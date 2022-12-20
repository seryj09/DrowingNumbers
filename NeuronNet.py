import numpy as np
import scipy.special


class NeuralNetwork:
    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, wih=None, who=None):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # коэффициент обучения
        self.lr = learningrate
        # Матрицы весовых коэффициентов связей wih (между входным и скрытым
        # слоями) и who (между скрытым и выходным слоями).
        # Весовые коэффициенты связей между узлом i и узлом j следующего слоя
        # обозначены как w_i_j:
        # w11 w21
        # w12 w22 и т.д.
        if isinstance(wih, type(None)):
            self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
            # Инициаллизировать веса можно и проще
            # self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
            # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        else:
            self.wih = np.array(wih)
            self.who = np.array(who)

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

    # тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # преобразовать список входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = np.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * np.dot((output_errors * final_outputs *
                                      (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # обновить весовые коэффициенты связей между входным и скрытым слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs *
                                      (1.0 - hidden_outputs)), np.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs