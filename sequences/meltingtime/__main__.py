#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Means for computing melting times:
"""
import click
from   ._computations import TransitionStats

@click.command()
@click.argument('sequence')
@click.argument('seqs', nargs = -1)
@click.option(
    "--force", '-f',
    default = 8.5,
    type    = float,
    help    = "force exerted on the hairpin"
)
def main(sequence, seqs, force): # type: ignore
    """
    Computes oligo transition stats.

    SEQUENCE and SEQS arguments work as follows:

    * starting with a '5-' is an oligo binding to the 5' strand
    * starting with a '3-' is an oligo binding to the 3' strand
    * following the '5-' or '3-' must come a number indicating the position
      relative to the hairpin
    * if only '5-' '3-' are indicated or if both these and numbers are
    missing, this is expected to be the hairpin's 5' (then 3') sequence

    Example:

    ```
    >> [PROGAM] CCCCTAGGGGATTACCC 5-3GATC 3-10TAAT 5-5AGGGA -f 8.5
    ```
    """
    cnf = TransitionStats(sequence, *seqs, force = force)
    out = cnf.statistics()
    print(cnf.strands.representation()+f"""
        
        trep:  {out[1]:.1f}
        K_on:  {out[2]:.1f}
        K_off: {out[3]:.1f}
        °T:    {out[4]:.1f}
    """.replace(" "*8, ""))

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
