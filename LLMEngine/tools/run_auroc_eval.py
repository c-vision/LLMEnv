#!/usr/bin/env python3
"""
Self-Consistency AUROC Evaluation Runner

Esegue valutazione completa: genera risposte modello, calcola risk via AI.CONSISTENCY,
confronta con ground truth, calcola AUROC.

Uso:
  python run_auroc_eval.py --model Qwen3.8-27B --fixture gsm8k --k 5 --temp 0.7 --max-samples 20
"""

import json
import subprocess
import sys
import os
import argparse
import tempfile
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Tuple
import re

# Paths
GMB_ROOT = Path('/Users/gaetano/Documents/Projects/GMBasic')
FIXTURE_DIR = GMB_ROOT / 'samples' / 'Fixtures' / 'ai'
GOLD_DIR = FIXTURE_DIR

def normalize_answer(text: str) -> str:
    """Normalizza risposta per confronto numerico"""
    # Estrae risposta finale
    match = re.search(r'####\s*([^\n]+)', text)
    if match:
        s = match.group(1).strip()
    else:
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        s = match.group(1).strip() if match else text.strip()
    
    s = s.lower().strip().replace(',', '').replace(' ', '')
    try:
        return str(float(s))
    except:
        return s

def load_gold(fixture_name: str) -> List[Dict]:
    """Carica fixture gold con ground truth"""
    gold_path = GOLD_DIR / f'{fixture_name}_300_gold.json' if fixture_name == 'gsm8k' else GOLD_DIR / f'{fixture_name}_gold.json'
    with open(gold_path, 'r') as f:
        data = json.load(f)
    return data['samples']

def build_bas_script(question: str, model: str, k: int, temp: float) -> str:
    """Costruisce codice BASIC per generare risposta + consistency check"""
    model_json = f'"{model}"'
    return f'''
IMPORT AI

' Imposta provider attivo
AI SET ACTIVE %CHAT, {model_json}

DIM question$ = {json.dumps(question)}
DIM answer$ = AI.ASK(question$)

' Self-consistency check
DIM risk = AI.CONSISTENCY(question$, {{"k": {k}, "temp": {temp}, "model": {model_json}}})
DIM detail$ = AI.CONSISTENCYDETAIL$()

PRINT "ANSWER:" + answer$
PRINT "RISK:" + STR(risk)
PRINT "DETAIL:" + detail$
'''

def run_bas_script(bas_code: str) -> Tuple[str, float, str]:
    """Esegue script BASIC e ritorna (answer, risk, detail_json)"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bas', delete=False) as f:
        f.write(bas_code)
        bas_file = f.name
    
    try:
        env = os.environ.copy()
        
        result = subprocess.run(
            ['dotnet', 'run', '--project', 'Executor', '--', bas_file],
            cwd=str(GMB_ROOT),
            capture_output=True, text=True, timeout=180,
            env=env
        )
        os.unlink(bas_file)
        
        if result.returncode != 0:
            return "", 0.5, "{}"
        
        answer = ""
        risk = 0.5
        detail = "{}"
        
        for line in result.stdout.split('\n'):
            if line.startswith('ANSWER:'):
                answer = line[7:].strip()
            elif line.startswith('RISK:'):
                try:
                    risk = float(line[5:].strip())
                except:
                    risk = 0.5
            elif line.startswith('DETAIL:'):
                detail = line[7:].strip()
        
        return answer, risk, detail
    
    except subprocess.TimeoutExpired:
        os.unlink(bas_file)
        return "", 0.5, "{}"
    except Exception as e:
        print(f"  Errore esecuzione: {e}")
        return "", 0.5, "{}"

def evaluate_fixture(fixture_name: str, model: str, k: int, temp: float, max_samples: int = None) -> Dict:
    """Esegue valutazione completa su un fixture"""
    samples = load_gold(fixture_name)
    
    print(f"Valutazione {fixture_name} con {model} (K={k}, T={temp})")
    
    risks = []
    labels = []
    
    for i, sample in enumerate(samples):
        if max_samples and i >= max_samples:
            break
            
        question = sample['question']
        ground_truth = sample['ground_truth']
        
        print(f"  [{i+1}/{min(len(samples), max_samples or len(samples))}] {question[:60]}...")
        
        bas_code = build_bas_script(question, model, k, temp)
        answer, risk, detail = run_bas_script(bas_code)
        
        if not answer:
            print(f"    Errore: risposta vuota")
            continue
        
        # Normalizza e confronta
        pred_norm = normalize_answer(answer)
        label = 1 if pred_norm == ground_truth else 0
        
        risks.append(risk)
        labels.append(label)
        
        print(f"    Risk: {risk:.4f}, Label: {label}, Answer: {answer[:80]}...")
    
    if len(labels) == 0:
        return {'fixture': fixture_name, 'auroc': None, 'n_samples': 0}
    
    # Calcola AUROC
    try:
        auroc = roc_auc_score(labels, risks)
    except:
        auroc = None
    
    # Calcola metriche aggiuntive
    accuracy = sum(labels) / len(labels) if labels else 0
    
    return {
        'fixture': fixture_name,
        'model': model,
        'k': k,
        'temp': temp,
        'n_samples': len(labels),
        'auroc': auroc,
        'accuracy': accuracy,
        'risks': risks,
        'labels': labels
    }

def main():
    parser = argparse.ArgumentParser(description='AUROC Evaluation for Self-Consistency')
    parser.add_argument('--model', required=True, help='Model name (es. Qwen3.8-27B-Flash-Next)')
    parser.add_argument('--fixture', choices=['gsm8k', 'math500', 'both'], default='both')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--max-samples', type=int, default=20)
    parser.add_argument('--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    print(f"=== Self-Consistency AUROC Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Fixture: {args.fixture}")
    print(f"K={args.k}, Temp={args.temp}, Max samples: {args.max_samples}")
    
    fixtures = ['gsm8k', 'math500'] if args.fixture == 'both' else [args.fixture]
    
    all_results = {}
    for fixture in fixtures:
        print(f"\n=== {fixture.upper()} ===")
        result = evaluate_fixture(fixture, args.model, args.k, args.temp, args.max_samples)
        all_results[fixture] = result
        
        if result.get('auroc') is not None:
            print(f"  AUROC: {result['auroc']:.4f}")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Samples: {result['n_samples']}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nRisultati salvati in {args.output}")

if __name__ == '__main__':
    main()