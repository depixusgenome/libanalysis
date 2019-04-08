#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Means for computing melting times:
"""
from ._computations import TransitionStats, configuration
from ._old          import OldStatesTransitions
from ._data         import nndata
if __name__ == '__main__':
    class _click:
        command  = staticmethod(lambda *_, **__: None)
        argument = staticmethod(lambda *_, **__: None)
        option   = staticmethod(lambda *_, **__: None)
    try:
        _click = __import__('click') # type: ignore
    except ImportError:
        pass

    @_click.command()
    @_click.argument('seq', nargs = -1, help = configuration.__doc__)
    @_click.option(
        "--force", '-f',
        default = 8.5,
        type    = float,
        help    = "force exerted on the hairpin"
    )
    def main(seq, force): # type: ignore
        "main function"
        cnf = configuration(*seq, force = force)
        out = cnf.statistics()
        print(f"""
            {str(cnf.strands)}

            trep:  {out[1]}
            k_on:  {out[2]}
            k_off: {out[3]}
            °T:    {out[4]}
        """)

    main() # pylint: disable=no-value-for-parameter
