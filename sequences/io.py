#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"All sequences-related stuff"
from pathlib import Path
from typing  import List, Union, Iterable, Iterator, Tuple, TextIO, Dict, cast
from utils   import initdefaults

def _read(stream:TextIO) -> Iterator[Tuple[str,str]]:
    "reads a path and yields pairs (name, sequence)"
    title = None
    seq   = ""
    ind   = 0
    first = True
    for line in stream:
        line = line.strip()
        if len(line) == 0:
            continue

        if line[0] == '#':
            continue

        if line[0] == '>':
            if len(seq):
                first = False
                yield ("hairpin %d" % (ind+1) if title is None else title, seq)
                ind += 1

            title = line[1:].strip()
            seq   = ''
            continue

        seq += line

    if len(seq):
        if first and title is None and getattr(stream, 'name', None) is not None:
            yield (Path(str(stream.name)).stem, seq)
        else:
            yield ("hairpin %d" % (ind+1) if title is None else title, seq)

def read(stream:Union[Path, str, Dict[str,str], TextIO]) -> Iterator[Tuple[str,str]]:
    "reads a path and yields pairs (name, sequence)"
    if isinstance(stream, dict):
        return cast(Iterator[Tuple[str, str]], stream.items())

    if isinstance(stream, Iterator):
        return cast(Iterator[Tuple[str, str]], stream)

    if isinstance(stream, str) and '/' not in stream and '.' not in stream:
        try:
            if not Path(stream).exists():
                return iter((('hairpin 1', stream),))
        except OSError:
            return iter((('hairpin 1', stream),))

    return _read(cast(TextIO, open(cast(str, stream))))

class LNAHairpin:
    """The theoretical sequence: full, target, references"""
    full:       str       = ""
    target:     str       = ""
    references: List[str] = []
    @initdefaults(frozenset(locals()),
                  path = lambda self, val: self.setfrompath(val))
    def __init__(self, **_):
        pass

    def setfrompath(self, file_sequence: Union[str, Path],
                    full:   str = 'full',
                    target: str = 'target',
                    references: Union[Iterable[str], str] = None):
        """file_sequence is the path of the fasta file
         format of the file:
            > full
            (...)cccatATTCGTATcGTcccat(...)
            > oligo
            cccat,tgtca
            > target
            TCGTAT
        """
        text = dict(read(file_sequence))

        self.full       = text.pop(full)
        self.target     = self.full if target is None else text.pop(target)

        if isinstance(references, str):
            references = [i.strip() for i in references.split(',')]
        itr = (text.values() if not references else (text[i] for i in references))
        self.references = sum((i.split(',') for i in itr), [])
