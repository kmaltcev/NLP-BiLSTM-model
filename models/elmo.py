from simple_elmo import ElmoModel


class Elmo(ElmoModel):
    def __init__(self):
        super().__init__()
        super().load("./212/", max_batch_size=40)
