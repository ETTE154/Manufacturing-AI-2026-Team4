"""Export integration cards from frozen result files without model fitting."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import json

OUT = Path(__file__).resolve().parent


def main():
    results = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    thresholds = json.loads((OUT / 'thresholds.json').read_text(encoding='utf-8'))
    candidates = json.loads((OUT / 'candidates.json').read_text(encoding='utf-8'))
    verification = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    data = {
        'schema_version': 1,
        'status': results['status'],
        'verification_status': verification['status'],
        'actual_fault_labels_available': False,
        'actual_precision_recall_available': False,
        'model_available_from': '2025-02-01T00:00:00',
        'chronological_protocol': {
            'training': {'start_inclusive': '2024-12-01T00:00:00', 'end_exclusive': '2025-01-01T00:00:00', 'roles': ['baseline', 'IF', 'AE', 'synthetic_classifiers']},
            'validation_calibration': {'start_inclusive': '2025-01-01T00:00:00', 'end_exclusive': '2025-02-01T00:00:00', 'roles': ['threshold_calibration', 'AE_epoch_selection', 'synthetic_classifier_selection']},
            'after_calibration_test': {'start_inclusive': '2025-02-01T00:00:00', 'end_exclusive': '2025-04-30T00:00:00', 'roles': ['unlabeled_sample_alarm_rates', 'independent_synthetic_test_bases']},
            'full_scan': {'start_inclusive': '2024-12-01T00:00:00', 'end_inclusive': '2025-04-30T00:00:00', 'training_and_calibration_periods_are_retrospective': True},
            'actual_sample_endpoints': results['chronological_splits']
        },
        'time_axis': {'field': 'localtime', 'format': '%Y%m%d%H%M%S', 'step_seconds': 5, 'timezone_conversion': None, 'Unix_timestamp_used_for_scoring': False},
        'models': {
            'contextual_rules': {'label': '모듈·시간대 조건 검사', 'role': 'six candidate patterns', 'threshold': 'January module/type 99.9th percentile; night rows only for night pattern', 'synthetic_test_metrics': results['synthetic_detection']['rules'], 'original_sample_alarm_fraction': results['original_test_alarm_rates']['rule_any_fraction'], 'full_source_scoring': True},
            'autoencoder': {'label': 'Autoencoder', 'role': 'pointwise multivariate reconstruction deviation', 'threshold': thresholds['ae'], 'training': results['autoencoder'], 'synthetic_test_metrics': results['synthetic_detection']['autoencoder'], 'original_sample_alarm_fraction': results['original_test_alarm_rates']['AE_alarm_fraction'], 'full_source_scoring': True, 'candidate_pattern': 'abnormal_pattern'},
            'isolation_forest': {'label': 'Isolation Forest', 'role': 'pointwise multivariate anomaly score', 'threshold': thresholds['iforest'], 'synthetic_test_metrics': results['synthetic_detection']['isolation_forest'], 'original_sample_alarm_fraction': results['original_test_alarm_rates']['IF_alarm_fraction'], 'full_source_scoring': False, 'coverage': results['full_scan']['IF_and_classifier_coverage']},
        },
        'types': [],
        'replay_events': [],
        'candidate_display_score': {'definition': 'rule_score/rule_threshold; abnormal_pattern=max(rule ratio,AE ratio)', 'threshold': 1.0, 'unit': 'dimensionless_exceedance_ratio', 'changes_frozen_model_thresholds': False},
        'synthetic_design': results['synthetic_design'],
        'selected_classifier': results['selected_classifier'],
        'classifier_selected_on': results['selected_on'],
        'model_files': results['model_files'],
        'source_sha256': results['full_scan']['source_sha256'],
        'source_rows': results['full_scan']['rows_read'],
        'limitations': results['limitations'] + [
            'Single points can trigger platform warnings; neither a warning nor sustained three-point interval confirms a fault.',
            'Original sample alarm fractions are not false-positive rates.',
            'Six-form synthetic evaluation measures explicit transformations; transfer to unknown real inserted forms is unverified.',
            'December/January replay is retrospective because the frozen detector becomes available on February1.',
            'Candidate score/threshold normalization is for display only; API rule scores retain native thresholds.'
        ],
        'paths': {name: str(OUT / name) for name in ['results.json', 'scan_summary.json', 'candidates.json', 'verification.json', 'README.md']}
    }
    for model_id, value in results['synthetic_classification'].items():
        data['models'][model_id] = {
            'label': '합성 형태 분류 XGBoost' if model_id == 'xgboost' else '합성 형태 분류 Random Forest',
            'role': 'seven-class synthetic transformation similarity',
            'selected': model_id == results['selected_classifier'],
            'validation_macro_f1': value['validation']['macro avg']['f1-score'],
            'test_macro_f1': value['test']['macro avg']['f1-score'],
            'test_accuracy': value['test']['accuracy'],
            'test_points': int(value['test']['macro avg']['support']),
            'probability_meaning': 'Synthetic transformation class similarity; not real-fault probability.',
            'full_source_scoring': False,
            'coverage': results['full_scan']['IF_and_classifier_coverage']
        }
    descriptions = {
        'load_imbalance': '상 전류와 모듈 평소 대비 전류 불균형 초과',
        'reactive_power_surge': 'Q·Q/P 증가와 역률 저하 보조 신호',
        'power_factor_drop': '최소·평균 역률 저하와 Q/P 증가 보조 신호',
        'abnormal_pattern': '강건 센서 편차·경험적 전력관계 편차 또는 AE 재구성 편차',
        'night_operation': '22–06시 모듈·시간대 기준 전력·전류 초과; operation=1만으로 경보하지 않음',
        'current_only_increase': '전류 증가와 경험적 전력관계 불일치'
    }
    labels = dict(zip(results['six_types'], ['부하 불균형', '무효전력 급증', '역률 급감', '비정상 패턴', '야간 이상가동', '전류만 증가']))
    for kind, count in results['full_scan']['point_and_event_counts'].items():
        data['types'].append({'id': kind, 'label': labels[kind], 'description': descriptions[kind], **count,
                              'threshold_by_module': {m: values[kind] for m, values in thresholds['rules'].items()}})
    for event in candidates['candidates']:
        if event['point_count'] < 3:
            continue
        data['replay_events'].append({k: event.get(k) for k in ['id', 'type', 'module_id', 'equipment', 'start', 'end', 'time', 'point_count', 'duration_seconds', 'score', 'threshold', 'source_kind', 'status', 'evaluation_scope']})
    (OUT / 'integration_metadata.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'exported', 'model_cards': len(data['models']), 'types': len(data['types']), 'replay_events': len(data['replay_events']), 'path': str(OUT / 'integration_metadata.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
