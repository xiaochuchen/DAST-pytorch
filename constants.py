
from pathlib import Path


WORK_DIR = Path(__file__).parent
DATA_DIR = WORK_DIR / "data"
IMDB_DIR = DATA_DIR / "filter_imdb"  # source
YELP_DIR = DATA_DIR / "yelp"  # target
AMZ_DIR = DATA_DIR / "amazon"  # target
IY_CLF_DIR = DATA_DIR / "iy-domainclf"
MIX_VOCAB = IY_CLF_DIR / "mixvocab.txt"
YELP_CLF_DIR = DATA_DIR / "yelp-styleclf"
TAR_VOCAB = YELP_CLF_DIR / "yelpvocab.txt"

OUT_DIR = DATA_DIR / "output"  # for models and sample results
LOG_DIR = WORK_DIR / "logs"  # for tensorboard logging
