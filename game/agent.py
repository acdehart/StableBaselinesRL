import random
from threading import Thread
from time import sleep, time


class paddle_boy:
    def chose_direction(self):
        step_size = 1
        # self.acceleration = random.choice([-1, 0, 1])
        self.acceleration = -self.ball_distance/10
        self.velocity += step_size*self.acceleration

    def choiceTimer(self):
        while True:
            self.calculate_score()
            if self.goFlag:
                self.chose_direction()
                self.goFlag = False
            else:
                sleep(0.25)
                self.goFlag = True

    def calculate_score(self):
        hits = self.hits
        run_time = time() - self.start_time
        dist_bias = - abs((self.ball_distance / 400) ** 2)
        score = hits ** 2 + run_time ** 2 + dist_bias
        # score = hits ** 2 + run_time ** 2
        # score = hits
        # score = dist_bias
        self.score = score

    def __init__(self):
        self.start_time = time()
        self.direction = "left"
        self.acceleration = 0
        self.velocity = 0
        self.ball_distance = 0
        self.goFlag = False
        self.hits = 0
        self.score = 0

        timer = Thread(target=self.choiceTimer)
        timer.start()