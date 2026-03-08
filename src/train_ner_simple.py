#!/usr/bin/env python3
"""
Script simple de entrenamiento NER para MEDDOCAN
Solo usa etiquetas MEDDOCAN del CSV oficial
"""

import json
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainerCallback
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from seqeval.scheme import IOB2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar estilo de gráficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Variable global para compute_metrics
id2label_global = None

# Historia de entrenamiento
training_history = {
    'train_loss': [],
    'eval_loss': [],
    'eval_f1': [],
    'eval_precision': [],
    'eval_recall': [],
    'eval_accuracy': [],
    'epochs': []
}


class MetricsCallback(TrainerCallback):
    """Callback para guardar historial de métricas."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Captura métricas durante entrenamiento."""
        if logs is not None:
            # Guardar loss de entrenamiento
            if 'loss' in logs:
                training_history['train_loss'].append({
                    'step': state.global_step,
                    'loss': logs['loss'],
                    'epoch': logs.get('epoch', 0)
                })
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Captura métricas de evaluación."""
        if metrics is not None:
            epoch = metrics.get('epoch', 0)
            training_history['epochs'].append(epoch)
            training_history['eval_loss'].append(metrics.get('eval_loss', 0))
            training_history['eval_f1'].append(metrics.get('eval_f1', 0))
            training_history['eval_precision'].append(metrics.get('eval_precision', 0))
            training_history['eval_recall'].append(metrics.get('eval_recall', 0))
            training_history['eval_accuracy'].append(metrics.get('eval_accuracy', 0))


def load_meddocan_labels(csv_path: Path) -> Set[str]:
    """Carga solo las etiquetas MEDDOCAN del CSV oficial."""
    logger.info(f"Cargando etiquetas MEDDOCAN desde: {csv_path}")
    
    meddocan_labels = set(['O'])
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['MEDDOCAN'].strip()
            if label and label != 'MEDDOCAN':
                meddocan_labels.add(f'B-{label}')
                meddocan_labels.add(f'I-{label}')
    
    logger.info(f"✅ Cargadas {(len(meddocan_labels) - 1) // 2} entidades MEDDOCAN")
    logger.info(f"   Total etiquetas IOB2: {len(meddocan_labels)}")
    
    return meddocan_labels


def create_label_mappings(meddocan_labels: Set[str]) -> tuple:
    """Crea mapeos label<->id para etiquetas MEDDOCAN."""
    sorted_labels = sorted(list(meddocan_labels))
    
    label2id = {label: i for i, label in enumerate(sorted_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    logger.info(f"Mapeo de etiquetas creado: {len(label2id)} etiquetas")
    
    return label2id, id2label


def filter_and_remap_labels(dataset_path: Path, meddocan_labels: Set[str], label2id: Dict[str, int]) -> List[Dict]:
    """
    Filtra dataset para solo incluir etiquetas MEDDOCAN y las remapea a los nuevos IDs.
    """
    logger.info(f"Filtrando dataset desde: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    
    # Cargar label2id original del modelo base
    base_config_path = Path("models/bsc-bio-ehr-es-meddocan/config.json")
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
        base_id2label = {int(k): v for k, v in base_config['id2label'].items()}
    
    filtered_data = []
    total_tokens = 0
    meddocan_tokens = 0
    
    for item in full_dataset:
        text = item['text']
        old_labels = item['labels']
        
        # Convertir IDs antiguos a etiquetas
        labels_str = [base_id2label.get(label_id, 'O') for label_id in old_labels]
        
        # Filtrar: convertir no-MEDDOCAN a 'O'
        filtered_labels_str = [label if label in meddocan_labels else 'O' for label in labels_str]
        
        # Remapear a nuevos IDs
        new_labels = [label2id[label] for label in filtered_labels_str]
        
        total_tokens += len(labels_str)
        meddocan_tokens += sum(1 for l in filtered_labels_str if l != 'O')
        
        filtered_data.append({
            'id': item['id'],
            'text': text,
            'labels': new_labels
        })
    
    logger.info(f"✅ Dataset filtrado: {len(filtered_data)} documentos")
    logger.info(f"   Tokens totales: {total_tokens}")
    logger.info(f"   Tokens MEDDOCAN: {meddocan_tokens} ({meddocan_tokens/total_tokens*100:.1f}%)")
    
    return filtered_data


def tokenize_and_align_labels(examples, tokenizer, label2id):
    """Tokeniza texto y alinea etiquetas con subtokens."""
    # Dividir texto en palabras (mismo método que se usó para crear las etiquetas)
    texts_as_words = [text.split() for text in examples["text"]]
    
    tokenized_inputs = tokenizer(
        texts_as_words,
        truncation=True,
        padding=False,
        max_length=512,
        is_split_into_words=True
    )
    
    labels = []
    for i, label_ids in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Token especial ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # Primera subtoken de una palabra
                if word_idx < len(label_ids):
                    aligned_labels.append(label_ids[word_idx])
                else:
                    aligned_labels.append(-100)
            else:
                # Subtokens siguientes de la misma palabra
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        
        labels.append(aligned_labels)
    
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs


def expand_classifier_weights(old_model, new_label2id, old_label2id):
    """
    Expande los pesos del clasificador manteniendo los pesos de etiquetas que coinciden.
    
    Args:
        old_model: Modelo con clasificador viejo
        new_label2id: Mapeo nuevo de etiquetas
        old_label2id: Mapeo viejo de etiquetas
    
    Returns:
        Nuevos pesos y bias del clasificador
    """
    old_classifier_weight = old_model.classifier.weight.data  # [old_num_labels, hidden_size]
    old_classifier_bias = old_model.classifier.bias.data  # [old_num_labels]
    
    hidden_size = old_classifier_weight.shape[1]
    new_num_labels = len(new_label2id)
    
    # Crear nuevos tensores con inicialización aleatoria
    new_weight = torch.randn(new_num_labels, hidden_size) * 0.02  # Inicialización similar a BERT
    new_bias = torch.zeros(new_num_labels)
    
    # Copiar pesos de etiquetas que coinciden
    labels_copiadas = 0
    labels_nuevas = 0
    
    for label, new_id in new_label2id.items():
        if label in old_label2id:
            old_id = old_label2id[label]
            # Copiar los pesos
            new_weight[new_id] = old_classifier_weight[old_id]
            new_bias[new_id] = old_classifier_bias[old_id]
            labels_copiadas += 1
        else:
            labels_nuevas += 1
    
    logger.info(f"✅ Clasificador expandido:")
    logger.info(f"   Etiquetas con pesos copiados: {labels_copiadas}/{new_num_labels}")
    logger.info(f"   Etiquetas nuevas (inicialización aleatoria): {labels_nuevas}/{new_num_labels}")
    
    return new_weight, new_bias


def compute_metrics(eval_pred):
    """Calcula métricas usando seqeval."""
    global id2label_global
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        pred_seq = []
        true_seq = []
        
        for p, l in zip(prediction, label):
            if l != -100:
                pred_seq.append(id2label_global[p])
                true_seq.append(id2label_global[l])
        
        if pred_seq:
            true_predictions.append(pred_seq)
            true_labels.append(true_seq)
    
    precision = precision_score(true_labels, true_predictions, mode='strict', scheme=IOB2, zero_division=0)
    recall = recall_score(true_labels, true_predictions, mode='strict', scheme=IOB2, zero_division=0)
    f1 = f1_score(true_labels, true_predictions, mode='strict', scheme=IOB2, zero_division=0)
    accuracy = accuracy_score(true_labels, true_predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def plot_training_history(output_dir: str):
    """Genera gráficas del historial de entrenamiento."""
    output_path = Path(output_dir) / "final" / "plots"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generando gráficas de entrenamiento...")
    
    # 1. Loss vs Epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if training_history['eval_loss']:
        epochs = training_history['epochs']
        ax.plot(epochs, training_history['eval_loss'], 'o-', label='Validation Loss', linewidth=2, markersize=8)
        
        # Añadir train loss (promedio por epoch)
        if training_history['train_loss']:
            train_epochs = []
            train_losses_by_epoch = {}
            for entry in training_history['train_loss']:
                epoch = int(entry['epoch'])
                if epoch not in train_losses_by_epoch:
                    train_losses_by_epoch[epoch] = []
                train_losses_by_epoch[epoch].append(entry['loss'])
            
            for epoch in sorted(train_losses_by_epoch.keys()):
                train_epochs.append(epoch)
            
            avg_train_loss = [np.mean(train_losses_by_epoch[e]) for e in sorted(train_losses_by_epoch.keys())]
            ax.plot(train_epochs, avg_train_loss, 's-', label='Training Loss (avg)', linewidth=2, markersize=8, alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Métricas vs Epochs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = training_history['epochs']
    
    # F1-Score
    axes[0, 0].plot(epochs, training_history['eval_f1'], 'o-', color='#2ecc71', linewidth=2, markersize=8)
    axes[0, 0].set_title('F1-Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Precision
    axes[0, 1].plot(epochs, training_history['eval_precision'], 'o-', color='#3498db', linewidth=2, markersize=8)
    axes[0, 1].set_title('Precision', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Recall
    axes[1, 0].plot(epochs, training_history['eval_recall'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[1, 0].set_title('Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Accuracy
    axes[1, 1].plot(epochs, training_history['eval_accuracy'], 'o-', color='#9b59b6', linewidth=2, markersize=8)
    axes[1, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.suptitle('Evaluation Metrics per Epoch', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path / "metrics_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Todas las métricas juntas
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epochs, training_history['eval_f1'], 'o-', label='F1-Score', linewidth=2, markersize=8)
    ax.plot(epochs, training_history['eval_precision'], 's-', label='Precision', linewidth=2, markersize=8)
    ax.plot(epochs, training_history['eval_recall'], '^-', label='Recall', linewidth=2, markersize=8)
    ax.plot(epochs, training_history['eval_accuracy'], 'd-', label='Accuracy', linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path / "all_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráficas guardadas en: {output_path}")


def save_training_report(output_dir: str, metrics: Dict, args):
    """Genera reporte detallado del entrenamiento."""
    output_path = Path(output_dir) / "final"
    
    # Guardar historial en JSON
    history_data = {
        'training_history': training_history,
        'final_metrics': metrics,
        'configuration': {
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'train_samples': 'saved_in_metrics',
            'val_samples': 'saved_in_metrics'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(history_data, f, indent=2)
    
    # Generar reporte en Markdown
    best_epoch = np.argmax(training_history['eval_f1']) + 1 if training_history['eval_f1'] else 0
    best_f1 = max(training_history['eval_f1']) if training_history['eval_f1'] else 0
    
    markdown_report = f"""# Reporte de Entrenamiento NER - MEDDOCAN

**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuración

- **Modelo base**: `{args.model}`
- **Epochs**: {args.epochs}
- **Batch size**: {args.batch_size}
- **Learning rate**: {args.learning_rate}

## Resultados Finales

| Métrica | Valor |
|---------|-------|
| **F1-Score** | **{metrics.get('eval_f1', 0):.4f}** |
| **Precision** | {metrics.get('eval_precision', 0):.4f} |
| **Recall** | {metrics.get('eval_recall', 0):.4f} |
| **Accuracy** | {metrics.get('eval_accuracy', 0):.4f} |

## Mejor Epoch

- **Epoch #{best_epoch}** alcanzó el mejor F1-Score: **{best_f1:.4f}**

## Progresión por Epoch

| Epoch | F1 | Precision | Recall | Accuracy | Val Loss |
|-------|-----|-----------|--------|----------|----------|
"""
    
    for i, epoch in enumerate(training_history['epochs']):
        markdown_report += f"| {int(epoch)} | {training_history['eval_f1'][i]:.4f} | "
        markdown_report += f"{training_history['eval_precision'][i]:.4f} | "
        markdown_report += f"{training_history['eval_recall'][i]:.4f} | "
        markdown_report += f"{training_history['eval_accuracy'][i]:.4f} | "
        markdown_report += f"{training_history['eval_loss'][i]:.4f} |\n"
    
    markdown_report += f"""
## Gráficas

- `plots/loss_curve.png` - Curva de pérdida (train/val)
- `plots/metrics_curves.png` - Métricas por epoch
- `plots/all_metrics.png` - Comparación de todas las métricas

## Archivos Generados

- `pytorch_model.bin` - Modelo entrenado
- `config.json` - Configuración del modelo
- `training_history.json` - Historial completo de entrenamiento
- `metrics.json` - Métricas finales
- `label_map.json` - Mapeo de etiquetas MEDDOCAN
"""
    
    with open(output_path / "TRAINING_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    logger.info(f"✅ Reporte guardado en: {output_path / 'TRAINING_REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento NER simple para MEDDOCAN')
    parser.add_argument('--model', type=str, default='PlanTL-GOB-ES/bsc-bio-ehr-es',
                       help='Modelo base a fine-tunear')
    parser.add_argument('--output_dir', type=str, default='models/ner-meddocan-simple',
                       help='Directorio de salida')
    parser.add_argument('--train_set', type=str, default='corpus/train_set.json',
                       help='Conjunto de entrenamiento')
    parser.add_argument('--val_set', type=str, default='corpus/validation_set.json',
                       help='Conjunto de validación')
    parser.add_argument('--meddocan_csv', type=str, default='etiquetas_anonimizacion_meddocan_carmenI.csv',
                       help='CSV con etiquetas MEDDOCAN')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Número de epochs (con early stopping alto)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (optimizado para RTX 5080)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--ignore_mismatched_sizes', action='store_true',
                       help='Ignorar diferencias de tamaño en el clasificador (para reentrenar modelos con diferentes etiquetas)')
    
    args = parser.parse_args()
    
    # 1. Cargar etiquetas MEDDOCAN
    meddocan_labels = load_meddocan_labels(Path(args.meddocan_csv))
    label2id, id2label = create_label_mappings(meddocan_labels)
    
    global id2label_global
    id2label_global = id2label
    
    # 2. Filtrar y cargar datasets
    logger.info("Filtrando datasets...")
    train_data = filter_and_remap_labels(Path(args.train_set), meddocan_labels, label2id)
    val_data = filter_and_remap_labels(Path(args.val_set), meddocan_labels, label2id)
    
    # 3. Crear HuggingFace datasets
    features = Features({
        'id': Value('string'),
        'text': Value('string'),
        'labels': Sequence(feature=Value('int64'))
    })
    
    train_dataset = Dataset.from_list(train_data, features=features)
    val_dataset = Dataset.from_list(val_data, features=features)
    
    # 4. Cargar tokenizer y modelo
    logger.info(f"Cargando modelo: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    
    if args.ignore_mismatched_sizes:
        # Expandir clasificador manteniendo pesos de etiquetas coincidentes
        logger.info("🔧 Modo de expansión de clasificador activado")
        
        # 1. Cargar modelo viejo para extraer pesos
        logger.info("   Paso 1: Cargando modelo original...")
        old_model = AutoModelForTokenClassification.from_pretrained(args.model)
        old_label2id = old_model.config.label2id
        old_id2label = old_model.config.id2label
        
        logger.info(f"   Modelo original: {len(old_label2id)} etiquetas")
        logger.info(f"   Modelo nuevo: {len(label2id)} etiquetas")
        
        # 2. Crear modelo nuevo con número correcto de etiquetas
        logger.info("   Paso 2: Creando modelo con nuevo número de etiquetas...")
        model = AutoModelForTokenClassification.from_pretrained(
            args.model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True  # Permite crear con diferente tamaño
        )
        
        # 3. Expandir clasificador copiando pesos de etiquetas coincidentes
        logger.info("   Paso 3: Expandiendo clasificador...")
        new_weight, new_bias = expand_classifier_weights(old_model, label2id, old_label2id)
        
        # 4. Asignar nuevos pesos al modelo
        model.classifier.weight.data = new_weight
        model.classifier.bias.data = new_bias
        
        logger.info("✅ Clasificador expandido exitosamente!")
        
        del old_model  # Liberar memoria
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # Carga normal (modelo base o sin expansión)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
    
    # 5. Tokenizar datasets
    logger.info("Tokenizando datasets...")
    train_dataset = train_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
        batched=True,
        remove_columns=['id', 'text']
    )
    
    val_dataset = val_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
        batched=True,
        remove_columns=['id', 'text']
    )
    
    # 6. Configurar entrenamiento
    # Optimizado para RTX 5080 (16GB VRAM) con PyTorch 2.9.0 + CUDA 13.0
    
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU detectada: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("⚠️ GPU no disponible, entrenando en CPU")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=0.05,  # Subido a 0.05 para mayor regularización L2 (evitar overfitting)
        warmup_ratio=0.1,   # 10% de warmup steps para arrancar suave el AdamW
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # RTX 5080 soporta BF16
        fp16=False,  # Usar BF16 en vez de FP16 para RTX 5080
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=False,  # No necesario con 16GB VRAM
        report_to="none",
        push_to_hub=False,
    )
    
    # 7. Entrenar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2), # Modificado a 2 (antes 3, insufiente para 5 epochs)
            MetricsCallback()
        ]
    )
    
    logger.info("="*60)
    logger.info("INICIANDO ENTRENAMIENTO")
    logger.info(f"  Modelo base: {args.model}")
    logger.info(f"  Etiquetas MEDDOCAN: {len(label2id)}")
    logger.info(f"  Ejemplos train: {len(train_dataset)}")
    logger.info(f"  Ejemplos val: {len(val_dataset)}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info("="*60)
    
    trainer.train()
    
    # 8. Guardar modelo final
    logger.info("Guardando modelo final...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    
    # Guardar label map
    with open(f"{args.output_dir}/final/label_map.json", 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
    
    # 9. Evaluar
    logger.info("Evaluando modelo final...")
    metrics = trainer.evaluate()
    
    logger.info("="*60)
    logger.info("RESULTADOS FINALES")
    logger.info(f"  F1-Score: {metrics.get('eval_f1', 0):.4f}")
    logger.info(f"  Precision: {metrics.get('eval_precision', 0):.4f}")
    logger.info(f"  Recall: {metrics.get('eval_recall', 0):.4f}")
    logger.info(f"  Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
    logger.info("="*60)
    
    # Guardar métricas
    with open(f"{args.output_dir}/final/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 10. Generar visualizaciones y reportes
    logger.info("\n" + "="*60)
    logger.info("GENERANDO VISUALIZACIONES Y REPORTES")
    logger.info("="*60)
    
    plot_training_history(args.output_dir)
    save_training_report(args.output_dir, metrics, args)
    
    logger.info("\n" + "="*60)
    logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    logger.info("="*60)
    logger.info(f"📁 Modelo guardado en: {args.output_dir}/final")
    logger.info(f"📊 Gráficas en: {args.output_dir}/final/plots")
    logger.info(f"📄 Reporte en: {args.output_dir}/final/TRAINING_REPORT.md")
    logger.info("="*60)


if __name__ == "__main__":
    main()

