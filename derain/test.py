try:
    # run in smart_derain.py
    from derain import Ui_DerainForm
    from video_box import VideoBox
    from util.utils import *
    import train
except ImportError:
    # run in smartcabin.py
    from derain.derain import Ui_DerainForm
    from derain.video_box import VideoBox
    from derain.util.utils import *
    import derain.train