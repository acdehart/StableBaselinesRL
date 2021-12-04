import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

Y_MAX = 10
Y_MIN = -10


def get_list_of_scores(classroom):
    scores = []
    for student in classroom:
        scores.append(get_modified_score(student.score))
    return scores


def get_list_of_names(classroom):
    names = []
    for student in classroom:
        names.append(str(student.id))
    return names


def get_modified_score(score):
    return (score/2 - 1)*100


def sort_students(students, length):
    temp_students = students[0:length]
    sorted_students = []
    while len(temp_students) is not 0:
        sorted_students.append(get_smallest(temp_students))
    return sorted_students


def get_smallest(students):
    current_student = students[0]
    for j, student in enumerate(students):
        if current_student.id > student.id:
            current_student = student
    students.remove(current_student)
    return current_student


class ClassPlotManager:

    def __init__(self, classroom):
        self.names = []
        self.scores = []
        self.class_averages = []
        self.generation = 0
        self.classroom = classroom
        self.students = classroom.students
        self.old_students = []
        self.colors = ['blue'] * len(self.students)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
        self.fig.set_size_inches(9, 4)

    def plot_live(self):
        self.ax1.cla()
        self.reset_colors()
        self.students = self.classroom.students
        self.animate()
        self.ax1.bar(self.names, self.scores, color=self.colors)
        self.ax1.set_title("Generation " + str(self.generation))
        self.ax1.set(xlabel='Student ID', ylabel='Scores (%)')
        self.ax1.set_ylim(Y_MIN, Y_MAX)
        self.show_scores()
        plt.show(block=False)
        plt.pause(0.5)
        self.old_students = self.names

    def set_second_plot(self):
        self.class_averages.append(self.classroom.class_average)
        self.ax2.cla()
        self.ax2.set_title("Class Averages")
        self.ax2.set(xlabel='Generation', ylabel='Average Scores')
        self.ax2.set_xlim(0, 10)
        self.ax2.yaxis.tick_right()
        self.ax2.set_ylim(2.03, 2.08)
        x = range(len(self.class_averages))
        y = self.class_averages
        self.ax2.plot(x, y, '--bo')

    def plot_get_result(self):
        self.ax1.cla()
        self.students = self.classroom.students
        self.get_gen_result()
        self.ax1.bar(self.old_students, self.scores, color=self.colors)
        self.ax1.set_title("Generation " + str(self.generation))
        self.ax1.set(xlabel='Student ID', ylabel='Scores (%)')
        self.ax1.set_ylim(Y_MIN, Y_MAX)
        self.show_scores()
        self.set_second_plot()
        plt.show(block=False)
        plt.pause(0.5)
        self.generation += 1

    def reset_colors(self):
        self.colors = ['blue'] * len(self.students)

    def show_scores(self):
        for i, v in enumerate(self.scores):
            self.ax1.text(i, Y_MAX-0.8, str(round(v, 2)), ha='center', va='center', color=self.colors[i])

    def animate(self):
        students = sort_students(self.students, len(self.students))
        self.names = get_list_of_names(students)
        self.scores = get_list_of_scores(students)

    def get_gen_result(self):
        self.names = get_list_of_names(self.students)
        for index, name in enumerate(self.old_students):
            if name not in self.names:
                self.colors[index] = 'red'
