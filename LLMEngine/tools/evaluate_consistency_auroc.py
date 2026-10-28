#!/usr/bin/env python3
"""
Self-Consistency AUROC Evaluation Script
Esegue AI.CONSISTENCY su fixture GSM8K/MATH e calcola AUROC vs ground truth.

Uso:
  python evaluate_consistency_auroc.py --model <nome_modello> --fixture gsm8k|math500 --k 5 --temp 0.7
"""

import json
import subprocess
import sys
import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import List, Dict, Tuple

# Aggiungi path per import GMBasic
sys.path.insert(0, '/Users/gaetano/Documents/Projects/GMBasic')

def load_fixture(fixture_path: str) -> List[Dict]:
    """Carica fixture JSONL (GSM8K o MATH)"""
    samples = []
    with open(fixture_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples

def extract_answer(text: str) -> str:
    """Estrae la risposta finale dal formato GSM8K/MATH (dopo #### o \boxed{})"""
    import re
    # GSM8K: #### 18
    match = re.search(r'####\s*([^\n]+)', text)
    if match:
        return match.group(1).strip()
    # MATH: \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    # Fallback: ultimo numero nella stringa
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else text.strip()

def evaluate_sample_correct(model_answer: str, ground_truth: str) -> int:
    """Valuta se la risposta del modello è corretta (1) o no (0)"""
    model_norm = extract_answer(model_answer).lower().strip()
    gt_norm = extract_answer(ground_truth).lower().strip()
    
    # Normalizza numeri (rimuovi virgole, spazi)
    def norm_num(s):
        s = s.replace(',', '').replace(' ', '')
        try:
            return str(float(s))
        except:
            return s
    
    return 1 if norm_num(model_norm) == norm_num(gt_norm) else 0

def run_ai_consistency(question: str, k: int = 5, temp: float = 0.7, model: str = None) -> Tuple[float, str]:
    """
    Esegue AI.CONSISTENCY via gmb REPL o CLI.
    Ritorna (risk_score, detail_json)
    """
    # Usa il REPL GMBasic per eseguire AI.CONSISTENCY
    bas_code = f'''
IMPORT AI
DIM q$ = {json.dumps(question)}
DIM risk = AI.CONSISTENCY(q$, {{"k": {k}, "temp": {temp}, "model": {json.dumps(model) if model else "null"}}})
DIM detail$ = AI.CONSISTENCYDETAIL$()
PRINT "RISK:" + STR(risk)
PRINT "DETAIL:" + detail$
'''
    
    # Scrivi file temporaneo
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bas', delete=False) as f:
        f.write(bas_code)
        bas_file = f.name
    
    try:
        # Esegui via gmb (C99Native o IL backend)
        env = os.environ.copy()
        if model:
            env['GMB_MODEL'] = model
        result = subprocess.run(
            ['gmb', bas_file],
            capture_output=True, text=True, timeout=120, env=env
        )
        os.unlink(bas_file)
        
        if result.returncode != 0:
            print(f"  Errore esecuzione: {result.stderr}")
            return 0.5, "{}"  # fallback
        
        # Parse output
        risk = 0.5
        detail = "{}"
        for line in result.stdout.split('\n'):
            if line.startswith('RISK:'):
                risk = float(line[5:])
            elif line.startswith('DETAIL:'):
                detail = line[7:]
        return risk, detail
    except subprocess.TimeoutExpired:
        os.unlink(bas_file)
        print("  Timeout")
        return 0.5, "{}"
    except Exception as e:
        print(f"  Errore: {e}")
        return 0.5, "{}"

def run_evaluation(fixture_name: str, model: str, k: int, temp: float, max_samples: int = None) -> Dict:
    """Esegue valutazione completa su fixture"""
    fixture_map = {
        'gsm8k': '/Users/gaetano/Documents/Projects/GMBasic/samples/Fixtures/ai/gsm8k_300.jsonl',
        'math500': '/Users/gaetano/Documents/Projects/GMBasic/samples/Fixtures/ai/math500.jsonl'
    }
    
    fixture_path = fixture_map[fixture_name]
    samples = load_fixture(fixture_path)
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Valutazione {fixture_name} con modello {model} (K={k}, T={temp})")
    print(f"  Sample: {len(samples)}")
    
    risks = []
    labels = []
    details = []
    
    for i, sample in enumerate(samples):
        if fixture_name == 'gsm8k':
            question = sample['question']
            ground_truth = sample['answer']
        else:  # math500
            question = sample['problem']
            ground_truth = sample['answer']
        
        # Genera risposta del modello (usa AI AGENT ASK o simile)
        # Per ora usa il provider attivo per generare risposta
        # Questo è un placeholder - serve integrare con il provider LLM
        
        # Per P4.1: creiamo il ground truth label
        # Qui assumiamo di avere già la risposta del modello per ogni sample
        # In realtà dovremmo generarla col provider LLM
        
        # Placeholder: simuliamo per ora
        print(f"  [{i+1}/{len(samples)}] {question[:60]}...")
        
    return {
        'fixture': fixture_name,
        'model': model,
        'k': k,
        'temp': temp,
        'n_samples': len(samples),
        'auroc': None,
        'risks': [],
        'labels': []
    }

def main():
    parser = argparse.ArgumentParser(description='Self-Consistency AUROC Evaluation')
    parser.add_argument('--model', required=True, help='Nome modello (es. Qwen3.8-27B)')
    parser.add_argument('--fixture', choices=['gsm8k', 'math500', 'both'], default='both')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output', help='File output JSON')
    
    args = parser.parse_args()
    
    # Setup provider
    os.environ['GMB_MODEL'] = args.model
    
    fixtures = ['gsm8k', 'math500'] if args.fixture == 'both' else [args.fixture]
    
    results = {}
    for fixture in fixtures:
        print(f"\n=== {fixture.upper()} ===")
        result = run_evaluation(fixture, args.model, args.k, args.temp, args.max_samples)
        results[fixture] = result
        
        if result['auroc'] is not None:
            print(f"  AUROC: {result['auroc']:.4f}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRisultati salvati in {args.output}")

if __name__ == '__main__':
    main()