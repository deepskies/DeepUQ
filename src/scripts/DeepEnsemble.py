import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transferring data from embargo butler to another butler"
    )

    # at least one arg in dataId needed for 'where' clause.
    parser.add_argument(
        "fromrepo",
        type=str,
        nargs="?",
        default="/repo/embargo",
        help="Butler Repository path from which data is transferred. \
            Input str. Default = '/repo/embargo'",
    )
    parser.add_argument(
        "torepo",
        type=str,
        help="Repository to which data is transferred. Input str",
    )
    parser.add_argument(
        "instrument",
        type=str,
        nargs="?",
        default="LATISS",
        help="Instrument. Input str",
    )
    parser.add_argument(
        "--embargohours",
        type=float,
        required=False,
        default=80.0,
        help="Embargo time period in hours. Input float",
    )
    parser.add_argument(
        "--datasettype",
        required=False,
        nargs="+",
        # default=[]
        help="Dataset type. Input list or str",
    )
    parser.add_argument(
        "--collections",
        # type=str,
        nargs="+",
        required=False,
        default="LATISS/raw/all",
        help="Data Collections. Input list or str",
    )
    parser.add_argument(
        "--nowtime",
        type=str,
        required=False,
        default="now",
        help="Now time in (ISO, TAI timescale). If left blank it will \
                        use astropy.time.Time.now.",
    )
    parser.add_argument(
        "--move",
        type=str,
        required=False,
        default="False",
        help="Copies if False, deletes original if True",
    )
    parser.add_argument(
        "--log",
        type=str,
        required=False,
        default="False",
        help="No logging if False, longlog if True",
    )
    parser.add_argument(
        "--desturiprefix",
        type=str,
        required=False,
        default="False",
        help="Define dest uri if you need to run ingest for raws",
    )
    return parser.parse_args()


if __name__ == "__main__":
    namespace = parse_args()
    # Define embargo and destination butler
    # If move is true, then you'll need write
    # permissions from the fromrepo (embargo)
    dest_butler = namespace.torepo
    if namespace.log == "True":
        # CliLog.initLog(longlog=True)
        logger = logging.getLogger("lsst.transfer.embargo")
        logger.info("from path: %s", namespace.fromrepo)
        logger.info("to path: %s", namespace.torepo)
