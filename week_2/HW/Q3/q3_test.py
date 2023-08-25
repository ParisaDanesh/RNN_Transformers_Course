import numpy as np
from tensorflow import keras
from keras import layers
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model = keras.models.load_model("./q3_method_1.keras")

pos_text = ("I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I "
            "was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I "
            "was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben "
            "Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. "
            "This one did exactly that. The entire theater (which was sold out) was overcome by laughter during "
            "the first half of the movie, and were moved to tears during the second half. While exiting the "
            "theater I not only saw many women in tears, but many full grown men as well, trying desperately not "
            "to let anyone see them crying. This movie was great, and I suggest that you go see it before you "
            "judge.")

neg_text = ("Some may go for a film like this but I most assuredly did not. A college professor, David Norwell, "
            "suddenly gets a yen for adoption. He pretty much takes the first child offered, a bad choice named Adam. "
            "As it turns out Adam doesn't have both oars in the water which, almost immediately, causes untold stress "
            "and turmoil for Dr. Norwell. This sob story drolly played out with one problem after another, "
            "all centered around Adam's inabilities and seizures. Why Norwell wanted to complicate his life with an "
            "unknown factor like an adoptive child was never explained. Along the way the good doctor managed to "
            "attract a wifey to share in all the hell the little one was dishing out. Personally, I think both of "
            "them were one beer short of a sixpack. Bypass this yawner.")

predictions = model.predict(np.array([pos_text]))
print(predictions[0])
