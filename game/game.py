from tkinter import *
import time
import random

from game_objects import Ball, Paddle, Bricks
canvas = None


class ball_game:
    def __init__(self, agent, show=False):
        root = Tk()
        root.title("Bounce")

        root.geometry(f"500x570+{self.get_window_displacement(show)}+0")
        root.resizable(0, 0)
        root.wm_attributes("-topmost", 1)
        canvas = Canvas(root, width=500, height=500, bd=0, highlightthickness=0, highlightbackground="Red", bg="Black")
        canvas.pack(padx=10, pady=10)
        score = Label(height=50, width=80, text="Score: 00", font="Consolas 14 bold")
        score.pack(side="left")
        root.update()

        score.configure(text="Score: 00")
        canvas.delete("all")
        BALL_COLOR = ["red", "yellow", "white"]
        BRICK_COLOR = ["PeachPuff3", "dark slate gray", "rosy brown", "light goldenrod yellow", "turquoise3", "salmon",
                       "light steel blue", "dark khaki", "pale violet red", "orchid", "tan", "MistyRose2",
                       "DodgerBlue4", "wheat2", "RosyBrown2", "bisque3", "DarkSeaGreen1"]
        random.shuffle(BALL_COLOR)
        paddle = Paddle(canvas, "blue", agent)

        bricks = self.Lay_Brick(BRICK_COLOR, canvas)

        self.ball = Ball(canvas, BALL_COLOR[0], paddle, bricks, score)
        root.update_idletasks()
        root.update()

        self.LoopGame(agent, canvas, paddle, root)

        root.destroy()

    def get_window_displacement(self, show):
        if show:
            displacement = 0
        else:
            displacement = 3000
        return displacement

    def Lay_Brick(self, BRICK_COLOR, canvas):
        bricks = []
        for i in range(0, 5):
            b = []
            for j in range(0, 19):
                random.shuffle(BRICK_COLOR)
                tmp = Bricks(canvas, BRICK_COLOR[0])
                b.append(tmp)
            bricks.append(b)
        for i in range(0, 5):
            for j in range(0, 19):
                canvas.move(bricks[i][j].id, 25 * j, 25 * i)
        return bricks

    def LoopGame(self, agent, canvas, paddle, root):
        # time.sleep(1)
        # for r in range(500):
        while True:
            if paddle.pausec != 1:
                try:
                    canvas.delete(m)
                    del m
                except:
                    pass
                if not self.ball.bottom_hit:
                    # LIVE GAME
                    self.ball.draw()
                    agent.hits = self.ball.hit
                    self.update_ball_distance(agent, paddle)
                    paddle.draw()
                    root.update_idletasks()
                    root.update()
                    time.sleep(0.001)
                    if self.ball.hit == 95:
                        canvas.create_text(250, 250, text="YOU WON !!", fill="yellow", font="Consolas 24 ")
                        root.update_idletasks()
                        root.update()
                        playing = False
                        break
                else:
                    canvas.create_text(250, 250, text="GAME OVER!!", fill="red", font="Consolas 24 ")
                    self.ball.hit = self.ball.hit / 2
                    root.update_idletasks()
                    root.update()
                    playing = False
                    break
            else:
                try:
                    if m == None: pass
                except:
                    m = canvas.create_text(250, 250, text="PAUSE!!", fill="green", font="Consolas 24 ")
                root.update_idletasks()
                root.update()

    def update_ball_distance(self, agent, paddle):
        ball_pos = self.ball.canvas.coords(self.ball.id)
        ball_x = (ball_pos[0] + ball_pos[2]) / 2
        paddle_pos = paddle.canvas.coords(paddle.id)
        paddle_x = (paddle_pos[0] + paddle_pos[2]) / 2
        agent.ball_distance = paddle_x - ball_x


