"""Plot four immutable original intervals with every frozen diagnostic model.

Only this plots directory is written. SourceStore is used as the approved
read-only window API; no fitting, threshold updates, or source edits occur.
"""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
ROOT = OUT.parents[3]
APP = ROOT / 'demo_platform'
sys.path.insert(0, str(ROOT / 'output/analysis/.plot_deps'))
sys.path.insert(0, str(APP / 'backend'))
sys.path.insert(0, str(OUT.parent))

import json
import hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, dates as mdates, font_manager
from data_source import SourceStore, SOURCE
from detector import Detector, prepare, TYPE_IDS, TYPE_LABELS

torch.set_num_threads(4)
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
                     'svg.fonttype': 'path', 'savefig.facecolor': '#ffffff'})
COLORS = ['#2265b2', '#d57f17', '#2e8b57']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def stat(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def decorate(ax, times, event):
    start = pd.Timestamp(event['start'])
    end = pd.Timestamp(event['end'])
    # Three sample timestamps are ten seconds apart; fifteen seconds of bins.
    ax.axvspan(start, end + pd.Timedelta(seconds=5), color='#efb44b', alpha=.20,
               label='보관 후보의 5초 관측 구간')
    ax.axvline(pd.Timestamp(event['time']), color='#bd4141', linewidth=.8, alpha=.8)
    ax.set_xlim(times.iloc[0], times.iloc[-1])
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(axis='y', alpha=.20)


def title(event, suffix):
    weak = '약한 조건 후보' if event['type'] == 'power_factor_drop' else '미확정 이상 후보'
    return (f"원본 {weak} · {event['equipment']} · {event['id']}\n"
            f"{pd.Timestamp(event['time']).strftime('%Y-%m-%d %H:%M:%S')} 최대점 ±3분 | {suffix}")


def save(fig, stem, note):
    fig.text(.015, .009, note, fontsize=8, color='#555555', va='bottom')
    fig.tight_layout(rect=(0, .045, 1, .92))
    paths = {}
    for ext in ['png', 'svg']:
        path = OUT / f'{stem}.{ext}'
        fig.savefig(path, dpi=180)
        paths[ext] = str(path)
    plt.close(fig)
    return paths


def sensor_figure(frame, event):
    fig, axes = plt.subplots(3, 1, figsize=(13.4, 8.6), sharex=True)
    fig.suptitle(title(event, '원센서'), fontsize=13)
    t = frame['dt']
    axes[0].plot(t, frame.activePower, color='#2265b2', linewidth=1.3, marker='.', markersize=3)
    axes[0].set_ylabel('유효전력\n원자료 (가이드 kW)')
    axes[0].set_title('유효전력', loc='left', fontsize=10)
    for phase, color in zip('RST', COLORS):
        axes[1].plot(t, frame[f'powerFactor{phase}'], color=color, linewidth=1.05,
                     marker='.', markersize=2.5, label=f'{phase}상')
        axes[2].plot(t, frame[f'current{phase}'], color=color, linewidth=1.05,
                     marker='.', markersize=2.5, label=f'{phase}상')
    axes[1].set_ylabel('역률 (%)')
    axes[1].set_title('세 상의 역률', loc='left', fontsize=10)
    axes[2].set_ylabel('전류 (A)')
    axes[2].set_title('세 상의 전류', loc='left', fontsize=10)
    axes[2].set_xlabel('기록 localtime · 5초 간격')
    for ax in axes:
        decorate(ax, t, event)
    axes[1].legend(loc='upper right', ncol=3, fontsize=9)
    axes[2].legend(loc='upper right', ncol=3, fontsize=9)
    return save(fig, event['id'] + '_sensors',
                '원본을 변형하지 않음. 음영은 보관 후보 3시점, 붉은 선은 최대점. 전력 숫자 스케일·계측 범위는 불확실하며 확정 고장 정답 없음.')


def rule_figure(frame, scored, event):
    ratios = scored['rule_scores'] / np.maximum(scored['rule_thresholds'], 1e-12)
    fig, axes = plt.subplots(3, 2, figsize=(13.4, 8.6), sharex=True)
    fig.suptitle(title(event, '여섯 조건 검사'), fontsize=13)
    for j, ax in enumerate(axes.flat):
        ax.plot(frame['dt'], ratios[:, j], color='#2265b2', linewidth=1.15, marker='.', markersize=3)
        ax.axhline(1., color='#bd4141', linestyle='--', linewidth=1, label='1월 보정 경계 = 1')
        ax.set_title(TYPE_LABELS[j], loc='left', fontsize=10)
        ax.set_ylabel('조건 점수 / 모듈별 경계')
        ax.set_ylim(0, max(1.15, float(np.max(ratios[:, j])) * 1.12))
        decorate(ax, frame['dt'], event)
    for ax in axes[-1]:
        ax.set_xlabel('기록 localtime · 5초 간격')
    return save(fig, event['id'] + '_rules',
                '초과배수는 고장 확률이 아님. 12월 모듈·시간대 기준선 / 1월 99.9백분위 경계를 동결. 음영 밖 단일 초과도 미확정 후보.')


def ai_figure(frame, scored, event, which, threshold):
    model_label = {'ae': 'Autoencoder', 'iforest': 'Isolation Forest'}[which]
    ratio = scored[f'{which}_score'] / threshold
    fig, ax = plt.subplots(figsize=(13.4, 4.0))
    fig.suptitle(title(event, model_label), fontsize=13)
    ax.plot(frame['dt'], ratio, color='#2e8b57', linewidth=1.3, marker='.', markersize=3)
    ax.axhline(1, color='#bd4141', linestyle='--', linewidth=1.1, label=f'동결 경계 ({threshold:.6g}) → 1')
    ax.set_ylim(0, max(1.15, float(np.max(ratio)) * 1.12))
    ax.set_ylabel(f'{model_label} 점수 / 경계')
    ax.set_xlabel('기록 localtime · 5초 간격')
    decorate(ax, frame['dt'], event)
    ax.legend(loc='upper right', fontsize=9)
    return save(fig, event['id'] + '_' + which,
                '구간의 모든 원본 5초점에 저장 모델을 적용. 12월 학습 / 1월 경계 보정, 추가 학습 없음. 임계값 초과는 확정 고장 진단이 아님.')


def main():
    source_before = stat(SOURCE)
    index_path = APP / 'data/source_index.json'
    assert index_path.exists(), 'Existing original index required; this plot script never builds an index.'
    original_index = json.loads(index_path.read_text(encoding='utf-8'))
    assert original_index['bytes'] == source_before['bytes']
    assert original_index['mtime_ns'] == source_before['mtime_ns']
    index_hash_before = sha(index_path)
    model_paths = [OUT.parent / n for n in ['models.joblib', 'ae.pt', 'synthetic_xgb.json', 'baseline.json', 'thresholds.json']]
    model_hashes = {p.name: sha(p) for p in model_paths}
    cases = json.loads((OUT.parent / 'candidates.json').read_text(encoding='utf-8'))
    events = [e for e in cases['candidates'] if e['point_count'] >= 3]
    assert len(events) == 4
    detector = Detector()
    store = SourceStore()
    manifest = {'status': 'in_progress', 'data_kind': 'recorded_original', 'confirmed_faults': False,
                'source_sha256_from_completed_fullscan': cases['source_sha256'],
                'source': str(SOURCE), 'window_definition': 'inclusive peak-3minutes through peak+3minutes;73five-second timestamps',
                'frozen_scoring': True, 'extra_training_epochs': 0, 'class_probabilities_are_fault_probabilities': False,
                'figures_per_event': ['sensors_3panel', 'six_normalized_rules', 'normalized_autoencoder', 'normalized_isolation_forest'],
                'format_per_figure': ['png', 'svg'], 'events': []}
    try:
        for event in events:
            peak = pd.Timestamp(event['time'])
            start = peak - pd.Timedelta(minutes=3)
            end_inclusive = peak + pd.Timedelta(minutes=3)
            rows = store.window(start, end_inclusive + pd.Timedelta(seconds=5), [event['module_id']])
            frame = prepare(rows)
            assert len(frame) == 73
            assert frame.dt.iloc[0] == start and frame.dt.iloc[-1] == end_inclusive
            assert frame.dt.diff().dropna().eq(pd.Timedelta(seconds=5)).all()
            scored = detector.scores(frame, full_ai=True)
            peak_i = int(np.flatnonzero(frame.dt.eq(peak).to_numpy())[0])
            expected = event['ai']
            for which in ['ae', 'iforest']:
                assert np.isclose(scored[which + '_score'][peak_i], expected[which + '_score'], rtol=0, atol=1e-8)
            figs = {'sensors': sensor_figure(frame, event), 'rules': rule_figure(frame, scored, event),
                    'autoencoder': ai_figure(frame, scored, event, 'ae', detector.saved['thresholds']['ae']),
                    'isolation_forest': ai_figure(frame, scored, event, 'iforest', detector.saved['thresholds']['iforest'])}
            ratios = scored['rule_scores'] / np.maximum(scored['rule_thresholds'], 1e-12)
            # Scores are enough to reproduce figures; dense sensor windows remain in original indexed source.
            audit = pd.DataFrame({'time': frame.dt, **{kind + '_ratio': ratios[:, j] for j, kind in enumerate(TYPE_IDS)},
                                  'ae_score': scored['ae_score'], 'ae_ratio': scored['ae_score'] / detector.saved['thresholds']['ae'],
                                  'iforest_score': scored['iforest_score'], 'iforest_ratio': scored['iforest_score'] / detector.saved['thresholds']['iforest']})
            audit_path = OUT / (event['id'] + '_scores.csv')
            audit.to_csv(audit_path, index=False, float_format='%.17g')
            manifest['events'].append({'id': event['id'], 'type': event['type'], 'module_id': event['module_id'], 'equipment': event['equipment'],
                                       'source_kind': 'recorded', 'status': 'unconfirmed_candidate',
                                       'peak_time': peak.isoformat(), 'event_start': event['start'], 'event_end': event['end'],
                                       'window_start_inclusive': start.isoformat(), 'window_end_inclusive': end_inclusive.isoformat(),
                                       'original_rows_scored_all_models': len(frame), 'saved_peak_scores_reproduced': True,
                                       'weak_candidate': event['type'] == 'power_factor_drop', 'figures': figs,
                                       'scores_csv': str(audit_path), 'peak_scores': {'rules_ratio': {k: float(ratios[peak_i, j]) for j, k in enumerate(TYPE_IDS)},
                                           'ae_ratio': float(scored['ae_score'][peak_i] / detector.saved['thresholds']['ae']),
                                           'iforest_ratio': float(scored['iforest_score'][peak_i] / detector.saved['thresholds']['iforest'])}})
    finally:
        store.mm.close()
        store.file.close()
    assert stat(SOURCE) == source_before
    assert sha(index_path) == index_hash_before
    assert {p.name: sha(p) for p in model_paths} == model_hashes
    manifest.update({'status': 'passed', 'source_size_mtime_preserved': True, 'source_index_unchanged': True,
                     'models_and_thresholds_unchanged': True, 'model_sha256': model_hashes,
                     'figure_count': len(events) * 4, 'image_file_count': len(events) * 8,
                     'original_rows_scored_all_models': 73 * len(events),
                     'interpretation': 'Only three AE abnormal-pattern intervals and one weak PF-rule interval are shown; actual fault labels unavailable.'})
    output = OUT / 'plot_manifest.json'
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'figure_count': manifest['figure_count'], 'image_file_count': manifest['image_file_count'],
                      'rows': manifest['original_rows_scored_all_models'], 'manifest': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
