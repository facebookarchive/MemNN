#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import re
import sys
import time
from multiprocessing import Process, Queue, Condition, Value

parser = argparse.ArgumentParser(
    description='Generates multi-dictionary versions of queries for WikiMovies.'
)
parser.add_argument('input_file', type=str, nargs='+',
    help='name of a input file in memnns format')
parser.add_argument('-d', type=int, default=1,
    help='Number of extra dictionaries.')
parser.add_argument('-o', '--output_file', type=str,
    help='name of a input file in memnns format')
parser.add_argument('-n', type=int, help='Max number of examples to process.')
parser.add_argument('-e', '--entities', type=str,
    help='entities file (each line specifies ngrams to always chunk together)')
parser.add_argument('-t', '--num_threads', type=int, default=1,
    help='number of text-processing threads to use (automatically uses a ' +
        'thread for reading and a thread for writing)')
args = vars(parser.parse_args())

beg = time.time()

if args['output_file']:
    out = open(args['output_file'], 'w')
else:
    out = sys.stdout

ent_list = []
re_list = []
entities = {}
ent_rev = {}
if 'entities' in args:
    if args['output_file']:
        print('Processing entity file...')
    with open(args['entities']) as read:
        for l in read:
            l = l.strip()
            if len(l) > 0:
                ent_list.append(l)
    ent_list.sort(key=lambda x: -len(x))
    for i in range(len(ent_list)):
        k = ent_list[i]
        v = '__{}__'.format(i)
        entities[k] = v
        ent_rev[v] = k
    re_list = [
        (
            re.compile('\\b{}\\b'.format(re.escape(e))),
            '{}'.format(entities[e])
        ) for e in ent_list
    ]
else:
    args['all_windows'] = True

splitter = re.compile('\\b.*?\S.*?(?:\\b|$)')
q_out = Queue()


def process_example(ex):
    if '\t' not in ex:
        raise ValueError('Should be query/answer pairs')
    tabs = ex.split('\t')
    ex = tabs[0]
    if 'entities' in args:
        # replace entities with single tokens
        for r, v in re_list:
            ex = r.sub(v, ex)
    split = [t.strip() for t in splitter.findall(ex)]
    extras = []
    for i in range(1, args['d'] + 1):
        for s in split:
            extras.append('{}:{}'.format(i, s))
    join = (
        '1 ' +
        ' '.join(s for s in split) + ' ' +
        ' '.join(e for e in extras) + '\t' +
        '\t'.join(t for t in tabs[1:]) + '\n'
    )
    if 'entities' in args:
        # put entities back in
        skip_idx = 0
        while True:
            fst_idx = join.find('__', skip_idx)
            if fst_idx > 0:
                snd_idx = join.find('__', fst_idx + 2)
                k = join[fst_idx:snd_idx + 2]
                if k in ent_rev:
                    join = join.replace(k, ent_rev[k])
                else:
                    skip_idx = snd_idx + 2
            else:
                break
    q_out.put(join)


# multithreading code
finished = Condition()
queued_exs = Value('i', 0)
proced_exs = Value('i', 0)
# keep at most 100 examples ready per thread queued (to save memory)
q = Queue(args['num_threads'] * 100)


def load(ex):
    global queued_exs
    queued_exs.value += 1
    q.put(ex)


def run():
    while True:
        ex = q.get()
        process_example(ex)


def write():
    global proced_exs
    while True:
        output = q_out.get()
        out.write(output)
        with proced_exs.get_lock():
            proced_exs.value += 1
        if q_out.empty() and queued_exs.value - proced_exs.value == 0:
            out.flush()
            with finished:
                finished.notify_all()


threads = []
threads.append(Process(target=write))
for i in range(args['num_threads']):
    threads.append(Process(target=run))
for t in threads:
    t.start()

if args['output_file']:
    print('Executing with {} threads.'.format(args['num_threads']))

mid = time.time()

for f in args['input_file']:
    if args['output_file']:
        # output is free to print debug info
        print('Processing file {}...'.format(f))
    cnt_exs = 0
    with open(f) as read:
        for line in read:
            line = line.strip()
            if line == '':
                continue
            idx = int(line[:line.find(' ')])
            line = line[line.find(' ') + 1:]
            if idx != 1:
                raise ValueError('Expected all single-line examples (queries).')
            cnt_exs += 1
            load(line)
            if args['n'] is not None and cnt_exs >= args['n']:
                break

while queued_exs.value - proced_exs.value > 0:
    with finished:
        finished.wait()

out.close()

for t in threads:
    t.terminate()

fin = time.time()
if args['output_file']:
    print('Time processing entities: {} s'.format(round(mid - beg)))
    print('Time processing examples: {} s'.format(round(fin - mid)))
    print('Total time: {} s'.format(round(fin - beg)))
