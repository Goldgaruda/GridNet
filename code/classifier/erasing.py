import Augmentor
p = Augmentor.Pipeline("./data/erasing")
p.random_erasing(probability=1, rectangle_area=0.25)
p.sample(520)