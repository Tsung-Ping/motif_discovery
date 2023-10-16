# -*- coding: utf-8 -*-
"""
Created on Tue Feb  20  2023
@author: Tsung-Ping Chen
"""
# import time
import numpy as np
import os
import csv
from os.path import join as jpath
from os.path import isdir
# import matplotlib.pyplot as plt
import time

import pretty_midi
from SIA import *
from mir_eval.pattern import establishment_FPR, occurrence_FPR, three_layer_FPR

'''Baseline algorithms'''
'''https://github.com/wsgan001/repeated_pattern_discovery'''
'''@repeated_pattern_discovery-master'''
import sys
sys.path.insert(1, r'\repeated_pattern_discovery-master')
from dataset import Dataset
from vector import Vector
import new_algorithms
import orig_algorithms

'''Directory of the Beethoven motif dataset'''
'''@Beethoven_motif-main'''
csv_note_dir = r'\Beethoven_motif-main\csv_notes'
csv_label_dir = r'\Beethoven_motif-main\csv_label'
motif_midi_dir = r'\Beethoven_motif-main\motif_midi'

'''Directory of the experiment dataset'''
'''@experiment_datasets'''
baseline_note_dir = r'\experiment_datasets\TsungPings_test_motif_dataset'

'''directory of the JKUPDD dataset'''
jkupdd_data_dir = r'\JKUPDD\JKUPDD-noAudio-Aug2013\groundTruth'
jkupdd_corpus = ['bachBWV889Fg', 'beethovenOp2No1Mvt3', 'chopinOp24No4', 'gibbonsSilverSwan1612', 'mozartK282Mvt2']
jkupdd_notes_csv = ['wtc2f20.csv', 'sonata01-3.csv', 'mazurka24-4.csv', 'silverswan.csv', 'sonata04-2.csv']


def load_all_notes(filename):
    '''Load all notes from CSV file'''
    dt = [
        ('onset', np.float32),
        ('pitch', np.int32),
        # ('mPitch', np.int32),
        ('duration', np.float32),
        ('staff', np.int32),
        ('measure', np.int32),
        ('type', '<U4')
    ] # datatype

    # Format data as structured array
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        notes = np.array([tuple([x for i, x in enumerate(row) if i != 2]) for row in reader], dtype=dt)

    # Get unique notes irrespective of 'staffNum'
    _, unique_indices = np.unique(notes[['onset', 'pitch']], return_index=True)
    notes = notes[unique_indices]

    notes = notes[notes['duration'] > 0]
    return np.sort(notes, order=['onset', 'pitch'])


def load_all_motives_csv(filename):
    '''Load all motives from CSV file'''
    dt = [
        ('onset', np.float32),
        ('end', np.float32),
        ('type', '<U4'),
        ('measure', np.int32),
        ('start_beat', np.float32),
        ('duration', np.float32),
        ('track', np.int32),
        ('time_sig', '<U5'),
        ('measure_score', np.int32),
        ('onset_midi', np.float32),
        ('end_midi', np.float32),
    ] # datatype

    # Format data as structured array
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip header
        motives_csv = np.array([tuple(row) for row in reader], dtype=dt)
    print('number of motives =', motives_csv.size)
    return np.sort(motives_csv, order=['onset', 'end'])


def load_all_motives_midi(filename):
    '''Load all motives from MIDI file'''
    dt = [
        ('onset', np.float32),
        ('end', np.float32),
        ('pitch', np.int32),
    ] # datatype

    midi_data = pretty_midi.PrettyMIDI(filename)
    notes_midi = [
        np.array([(note.start, note.end, note.pitch) for note in instrument.notes], dtype=dt)
        for instrument in midi_data.instruments
    ] # [(notes in track i), ...]
    return notes_midi


def load_all_motives(filename_csv, filename_midi):
    '''Combine motif informations of CSV and MIDI files'''
    motives_csv = load_all_motives_csv(filename_csv)
    motives_midi = load_all_motives_midi(filename_midi)
    motives = {}
    max_n_notes = 0
    max_duration = 0
    for motif in motives_csv:
        type = motif['type']
        if type not in motives.keys():
            motives[type] = [] # create a motif type

        track = motif['track'] # track id
        onset_midi, end_midi = motif[['onset_midi', 'end_midi']] # onset and end in midi time
        onset_calibration = motif['onset_midi'] - motif['onset'] # onset calibration if pickup measure
        track_notes = motives_midi[track]
        cond = (track_notes['onset'] >= onset_midi) & (track_notes['onset'] < end_midi)
        motif_notes = track_notes[cond]

        if motif_notes.size > max_n_notes:
            max_n_notes = motif_notes.size
        if motif['end'] - motif['onset'] > max_duration:
            max_duration = motif['end'] - motif['onset']

        motif_notes['onset'] -= onset_calibration
        motif_notes['end'] -= onset_calibration
        motives[type].append(motif_notes)

    assert len([motif for types in motives.values() for motif in types]) == motives_csv.size
    print('number of motif types', len(motives.keys()))
    print('max_n_notes', max_n_notes)
    print('max_duration', max_duration)
    return motives


def main():
    print('******* proposed eval *******')

    all_P_est, all_R_est, all_F_est = [], [], []
    all_P_occ, all_R_occ, all_F_occ = [], [], []
    all_P_thr, all_R_thr, all_F_thr = [], [], []
    runtime = 0
    for i in range(1,33):
        piece = str(i).zfill(2)
        print('piece', piece)

        filename_notes = os.path.join(csv_note_dir, piece+'-1.csv')
        filename_csv = os.path.join(csv_label_dir, piece+'-1.csv')
        filename_midi = os.path.join(motif_midi_dir, piece+'-1.mid')
        notes = load_all_notes(filename_notes)
        motives = load_all_motives(filename_csv, filename_midi)

        # Convert motives to mir_eval format
        patterns_ref = [[list(occur[['onset', 'pitch']]) for occur in motif] for motif in motives.values()]

        start_time = time.time()
        patterns_est = find_motives(notes)
        runtime_one = time.time() - start_time
        runtime += runtime_one
        print('runtime_one %.4f' % runtime_one)

        # Evaluation
        elp = time.time()
        F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
        F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
        F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
        print('est P %.4f, R %.4f, F %.4f' % (P_est, R_est, F_est))
        print('occ P %.4f, R %.4f, F %.4f' % (P_occ, R_occ, F_occ))
        print('thr P %.4f, R %.4f, F %.4f' % (P_thr, R_thr, F_thr))
        print('elapsed time, eval %.2f sec' % (time.time() - elp))

        all_P_est.append(P_est)
        all_R_est.append(R_est)
        all_F_est.append(F_est)
        all_P_occ.append(P_occ)
        all_R_occ.append(R_occ)
        all_F_occ.append(F_occ)
        all_P_thr.append(P_thr)
        all_R_thr.append(R_thr)
        all_F_thr.append(F_thr)
        # exit()

    mean_P_est = np.mean(all_P_est)
    mean_R_est = np.mean(all_R_est)
    mean_F_est = np.mean(all_F_est)
    mean_P_occ = np.mean(all_P_occ)
    mean_R_occ = np.mean(all_R_occ)
    mean_F_occ = np.mean(all_F_occ)
    mean_P_thr = np.mean(all_P_thr)
    mean_R_thr = np.mean(all_R_thr)
    mean_F_thr = np.mean(all_F_thr)
    print('Mean_est P %.4f, R %.4f, F %.4f' % (mean_P_est, mean_R_est, mean_F_est))
    print('Mean_occ P %.4f, R %.4f, F %.4f' % (mean_P_occ, mean_R_occ, mean_F_occ))
    print('Mean_thr P %.4f, R %.4f, F %.4f' % (mean_P_thr, mean_R_thr, mean_F_thr))
    print('Runtime %.4f Averaged Runtime %.4f' % (runtime / 60, runtime / 1920))


def de_vec(vec_obj_list):
    return [tuple(v) for v in vec_obj_list]


def get_all_occurrences(tec):
    return [[tuple(point + translator) for point in tec.get_pattern()] for translator in tec.get_translators()]


def mtps_to_tecs(mtps, dataset):
    sorted_dataset = Dataset.sort_ascending(dataset)
    v, w = orig_algorithms.compute_vector_tables(sorted_dataset)
    ciss = [[sorted_dataset._vectors.index(point) for point in intersection] for diff_vec, intersection in mtps]
    mcps = [(mtp[1], cis) for mtp, cis in zip(mtps, ciss)]
    orig_algorithms.remove_trans_eq_mtps(mcps)
    tecs = orig_algorithms.compute_tecs_from_mcps(sorted_dataset, w, mcps)
    return tecs


def baseline_eval():
    print('******* baseline eval *******')
    all_P_est, all_R_est, all_F_est = [], [], []
    all_P_occ, all_R_occ, all_F_occ = [], [], []
    all_P_thr, all_R_thr, all_F_thr = [], [], []
    for i in range(1,33):
        piece = str(i).zfill(2)
        print('piece', piece)

        # filename_notes = os.path.join(csv_note_dir, piece+'-1.csv')
        # notes = load_all_notes(filename_notes)
        filename_csv = os.path.join(csv_label_dir, piece+'-1.csv')
        filename_midi = os.path.join(motif_midi_dir, piece+'-1.mid')
        motives = load_all_motives(filename_csv, filename_midi)

        # Convert motives to mir_eval format
        patterns_ref = [[list(occur[['onset', 'pitch']]) for occur in motif] for motif in motives.values()]

        # Read dataset
        filename_eval = os.path.join(baseline_note_dir, str(i) + '.csv')
        dataset = Dataset(filename_eval)
        print('len(dataset)', len(dataset))

        # Get all the occurrences of all the maximal repeated patterns in the dataset
        elp = time.time()
        '''Baseline algorithms'''
        # tecs = new_algorithms.siatechf(dataset, min_cr=2)
        # tecs = orig_algorithms.cosiatech(dataset)
        tecs = orig_algorithms.siatech_compress(dataset)
        # tecs = mtps_to_tecs(orig_algorithms.siar(dataset, r=1), dataset)
        # tecs, _ = orig_algorithms.forths_algorithm(dataset, c_min=2, sigma_min=1)
        print('elapsed time, tec %.2f sec' % (time.time() - elp))

        # Convert tecs to mir_eval format
        patterns_est = [get_all_occurrences(tec) for tec in tecs if len(tec.get_translators())]
        print('len(patterns_est)', len(patterns_est))

        # Evaluation
        elp = time.time()
        F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
        F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
        F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
        print('est P %.4f, R %.4f, F %.4f' % (P_est, R_est, F_est))
        print('occ P %.4f, R %.4f, F %.4f' % (P_occ, R_occ, F_occ))
        print('thr P %.4f, R %.4f, F %.4f' % (P_thr, R_thr, F_thr))
        print('elapsed time, eval %.2f sec' % (time.time() - elp))

        all_P_est.append(P_est)
        all_R_est.append(R_est)
        all_F_est.append(F_est)
        all_P_occ.append(P_occ)
        all_R_occ.append(R_occ)
        all_F_occ.append(F_occ)
        all_P_thr.append(P_thr)
        all_R_thr.append(R_thr)
        all_F_thr.append(F_thr)
        # exit()

    mean_P_est = np.mean(all_P_est)
    mean_R_est = np.mean(all_R_est)
    mean_F_est = np.mean(all_F_est)
    mean_P_occ = np.mean(all_P_occ)
    mean_R_occ = np.mean(all_R_occ)
    mean_F_occ = np.mean(all_F_occ)
    mean_P_thr = np.mean(all_P_thr)
    mean_R_thr = np.mean(all_R_thr)
    mean_F_thr = np.mean(all_F_thr)
    print('Mean_est P %.4f, R %.4f, F %.4f' % (mean_P_est, mean_R_est, mean_F_est))
    print('Mean_occ P %.4f, R %.4f, F %.4f' % (mean_P_occ, mean_R_occ, mean_F_occ))
    print('Mean_thr P %.4f, R %.4f, F %.4f' % (mean_P_thr, mean_R_thr, mean_F_thr))


def load_jkupdd_notes_csv(csv_dir):
    dt = [
        ('onset', np.float32),
        ('pitch', np.int32),
        ('duration', np.float32),
        ('staff', np.int32),
    ] # datatype

    # Format data as structured array
    with open(csv_dir, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        notes = np.array([tuple([float(x) for i, x in enumerate(row) if i != 2]) for row in reader], dtype=dt)

    # Get unique notes irrespective of 'staffNum'
    _, unique_indices = np.unique(notes[['onset', 'pitch']], return_index=True)
    notes = notes[unique_indices]
    print('deleted notes:', [i for i in range(notes.size) if i not in unique_indices])

    notes = notes[notes['duration'] > 0]
    return np.sort(notes, order=['onset', 'pitch'])


def load_jkupdd_patterns_csv(csv_dir):
    annotators_dir = [jpath(csv_dir, f) for f in os.listdir(csv_dir) if isdir(jpath(csv_dir, f))]
    # print(annotators_dir)
    patterns_dir = [
        jpath(annotator, f) for annotator in annotators_dir for f in os.listdir(annotator) if isdir(jpath(annotator, f))
    ]
    # print(patterns_dir)

    patterns = []
    for pattern_dir in patterns_dir:
        occurrences_csv = [
            jpath(pattern_dir, 'occurrences/csv', f)
            for f in os.listdir(jpath(pattern_dir, 'occurrences/csv')) if f.endswith('.csv')
        ]
        # print(occurrences_csv)

        pattern = []
        for occurrence_csv in occurrences_csv:
            with open(occurrence_csv, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                pattern.append([tuple([float(x) for x in row]) for row in reader])
        # print(pattern)
        patterns.append(list(pattern))

    print('number of patterns', len(patterns))
    [print('pattern %s with %d occurrences' % (chr(i+65), len(pattern)))for i, pattern in enumerate(patterns)]
    return patterns


def jkupdd_eval():

    all_P_est, all_R_est, all_F_est = [], [], []
    all_P_occ, all_R_occ, all_F_occ = [], [], []
    all_P_thr, all_R_thr, all_F_thr = [], [], []
    runtime = 0
    total_n_notes = 0
    for i in range(5):
        print('file %s' % jkupdd_notes_csv[i])
        note_csv_dir = jpath(jkupdd_data_dir, jkupdd_corpus[i], 'polyphonic\csv', jkupdd_notes_csv[i])
        pattern_csv_dir = jpath(jkupdd_data_dir, jkupdd_corpus[i], r'polyphonic\repeatedPatterns')
        # note_csv_dir = jpath(jkupdd_data_dir, jkupdd_corpus[i], 'polyphonic/csv', jkupdd_notes_csv[i])
        # pattern_csv_dir = jpath(jkupdd_data_dir, jkupdd_corpus[i], 'polyphonic/repeatedPatterns')
        notes = load_jkupdd_notes_csv(note_csv_dir)
        dataset = Dataset(note_csv_dir)
        dataset._vectors = [Vector(list(x)[:2]) for x in dataset]
        patterns_ref = load_jkupdd_patterns_csv(pattern_csv_dir)

        print(notes)
        total_n_notes += len(notes)
        continue

        start_time = time.time()
        # patterns_est = find_motives(notes) # proposed algorithm
        # tecs = new_algorithms.siatechf(dataset, min_cr=2)
        # tecs = orig_algorithms.cosiatech(dataset)
        tecs = orig_algorithms.siatech_compress(dataset)
        # Convert tecs to mir_eval format
        patterns_est = [get_all_occurrences(tec) for tec in tecs if len(tec.get_translators())]
        elp = time.time() - start_time
        runtime += elp
        print('elapsed time %.4f sec' % elp)
        # print('len(patterns_est)', len(patterns_est))
        # print(patterns_est)
        # exit()

        # Evaluation
        # elp = time.time()
        F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
        F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
        F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
        print('est P %.4f, R %.4f, F %.4f' % (P_est, R_est, F_est))
        print('occ P %.4f, R %.4f, F %.4f' % (P_occ, R_occ, F_occ))
        print('thr P %.4f, R %.4f, F %.4f' % (P_thr, R_thr, F_thr))
        # print('elapsed time, eval %.2f sec' % (time.time() - elp))

        all_P_est.append(P_est)
        all_R_est.append(R_est)
        all_F_est.append(F_est)
        all_P_occ.append(P_occ)
        all_R_occ.append(R_occ)
        all_F_occ.append(F_occ)
        all_P_thr.append(P_thr)
        all_R_thr.append(R_thr)
        all_F_thr.append(F_thr)
        # exit()

    print('avg notes %d' %(total_n_notes/5))
    mean_P_est = np.mean(all_P_est)
    mean_R_est = np.mean(all_R_est)
    mean_F_est = np.mean(all_F_est)
    mean_P_occ = np.mean(all_P_occ)
    mean_R_occ = np.mean(all_R_occ)
    mean_F_occ = np.mean(all_F_occ)
    mean_P_thr = np.mean(all_P_thr)
    mean_R_thr = np.mean(all_R_thr)
    mean_F_thr = np.mean(all_F_thr)
    print('Mean_est P %.4f, R %.4f, F %.4f' % (mean_P_est, mean_R_est, mean_F_est))
    print('Mean_occ P %.4f, R %.4f, F %.4f' % (mean_P_occ, mean_R_occ, mean_F_occ))
    print('Mean_thr P %.4f, R %.4f, F %.4f' % (mean_P_thr, mean_R_thr, mean_F_thr))
    print('Runtime %.4f min Averaged runtime %.4f min' % (runtime / 60, runtime / 300))



if __name__ == '__main__':
    '''
    main(): the proposed method on the motif dataset
    baseline_eval(): baseline algorithms on the motif dataset
    jkupdd_eval(): evaluation on the JKUPD dataset
    '''
    main()
    # baseline_eval()
    # jkupdd_eval()