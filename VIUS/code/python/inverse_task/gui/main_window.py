import sys
import numpy as np
import scipy.io
import pickle
from scipy.interpolate import CubicSpline, UnivariateSpline
import plotly.graph_objects as go
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QCheckBox, QComboBox,
                             QPushButton, QSlider, QLabel, QProgressBar,
                             QTabWidget, QDockWidget, QFileDialog, QMessageBox,
                             QSplitter, QTableWidget, QTableWidgetItem,
                             QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from pathlib import Path

# Добавляем корневую директорию проекта (родительскую по отношению к папке gui)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
# Импорт ваших модулей (путь может потребовать настройки)
from machine.machine3axis_exact import Machine3AxisExact_ODE
from machine.kinematic_model import KinematicModel
from machine.kinematics_base import MachineState


class ComputeThread(QThread):
    """Поток для выполнения пересчёта без блокировки GUI."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, kin_model, s_span, q0_free, fixed_funcs, fixed_indices,
                 tsn_func, mandrel_func, d_tsn, d_mandrel, s_eval,
                 alpha, method, step=None):
        super().__init__()
        self.kin_model = kin_model
        self.s_span = s_span
        self.q0_free = q0_free
        self.fixed_funcs = fixed_funcs
        self.fixed_indices = fixed_indices
        self.tsn_func = tsn_func
        self.mandrel_func = mandrel_func
        self.d_tsn = d_tsn
        self.d_mandrel = d_mandrel
        self.s_eval = s_eval
        self.alpha = alpha
        self.method = method
        self.step = step

    def run(self):
        try:
            if self.method == 'adaptive':
                result = self.kin_model.integrate_fixed(
                    s_span=self.s_span,
                    q0_free=self.q0_free,
                    fixed_funcs=self.fixed_funcs,
                    fixed_indices=self.fixed_indices,
                    tsn_func=self.tsn_func,
                    mandrel_func=self.mandrel_func,
                    d_tsn_func=self.d_tsn,
                    d_mandrel_func=self.d_mandrel,
                    s_eval=self.s_eval,
                    alpha=self.alpha
                )
            else:
                step = self.step if self.step else 1.0
                result = self.kin_model.integrate_fixed_step(
                    s_span=self.s_span,
                    q0_free=self.q0_free,
                    fixed_funcs=self.fixed_funcs,
                    fixed_indices=self.fixed_indices,
                    tsn_func=self.tsn_func,
                    mandrel_func=self.mandrel_func,
                    d_tsn_func=self.d_tsn,
                    d_mandrel_func=self.d_mandrel,
                    step=step,
                    s_eval=self.s_eval,
                    alpha=self.alpha
                )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class RefinementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Совместная коррекция развёртки")
        self.setGeometry(100, 100, 1400, 900)

        # Данные
        self.data = None
        self.machine = None
        self.kin_model = None
        self.axes_names = []
        self.axes_units = []
        self.s_array = None
        self.q_orig = None          # массив (N, n_axes)
        self.tsn_pts = None
        self.mandrel_pts = None
        self.z_offset = None

        # Виджеты, создаваемые динамически
        self.plot_canvases = []      # список (canvas, axis_index)
        self.axis_checkboxes = []    # список чекбоксов для фиксации
        self.smooth_sliders = []     # для каждой фиксированной оси – свой слайдер? упростим: один общий для всех

        self._init_ui()

    def _init_ui(self):
        # Центральные вкладки
        self.central_tabs = QTabWidget()
        self.setCentralWidget(self.central_tabs)
        self.plots_tab = QWidget()
        self.central_tabs.addTab(self.plots_tab, "Графики")
        self.scene3d_tab = QWidget()
        self.central_tabs.addTab(self.scene3d_tab, "3D вид")
        self.table_tab = QWidget()
        self.central_tabs.addTab(self.table_tab, "Таблица")

        # Левая панель
        self.control_dock = QDockWidget("Управление", self)
        self.control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.control_dock)
        self._setup_control_panel()

        # Статусная строка с прогрессом
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Инициализация вкладок (заполнятся после загрузки)
        self.plots_layout = QVBoxLayout(self.plots_tab)
        self.table_widget = QTableWidget()
        QVBoxLayout(self.table_tab).addWidget(self.table_widget)

        # 3D вид – используем QWebEngineView для plotly
        self.web_view = QWebEngineView()
        QVBoxLayout(self.scene3d_tab).addWidget(self.web_view)

        self.run_btn.setEnabled(False)

    def _setup_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Загрузка данных
        load_group = QGroupBox("Загрузка")
        load_layout = QVBoxLayout()
        self.load_data_btn = QPushButton("Загрузить kinematics_results_full.mat")
        self.load_machine_btn = QPushButton("Загрузить параметры станка (machine_params.pkl)")
        load_layout.addWidget(self.load_data_btn)
        load_layout.addWidget(self.load_machine_btn)
        load_group.setLayout(load_layout)

        # Выбор осей для фиксации
        axes_group = QGroupBox("Фиксируемые оси")
        self.axes_container = QWidget()
        self.axes_layout = QVBoxLayout(self.axes_container)
        axes_group.setLayout(QVBoxLayout())
        axes_group.layout().addWidget(self.axes_container)

        # Параметры сглаживания (общие для всех)
        smooth_group = QGroupBox("Сглаживание")
        smooth_layout = QVBoxLayout()
        self.smooth_param_slider = QSlider(Qt.Horizontal)
        self.smooth_param_slider.setRange(0, 10000)
        self.smooth_param_slider.setValue(1000)
        self.smooth_label = QLabel("Параметр s: 1000")
        self.smooth_param_slider.valueChanged.connect(
            lambda v: self.smooth_label.setText(f"Параметр s: {v}"))
        smooth_layout.addWidget(self.smooth_label)
        smooth_layout.addWidget(self.smooth_param_slider)
        smooth_group.setLayout(smooth_layout)

        # Решатель
        solver_group = QGroupBox("Решатель")
        solver_layout = QVBoxLayout()
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["Адаптивный (solve_ivp, RK45)", "Постоянный шаг (RK4)"])
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.01, 100.0)
        self.step_spin.setValue(1.0)
        self.step_spin.setEnabled(False)
        self.solver_combo.currentIndexChanged.connect(
            lambda idx: self.step_spin.setEnabled(idx == 1))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 10.0)
        self.alpha_spin.setSingleStep(0.5)
        self.alpha_spin.setValue(2.0)
        solver_layout.addWidget(QLabel("Метод:"))
        solver_layout.addWidget(self.solver_combo)
        solver_layout.addWidget(QLabel("Шаг (мм, только для постоянного):"))
        solver_layout.addWidget(self.step_spin)
        solver_layout.addWidget(QLabel("alpha:"))
        solver_layout.addWidget(self.alpha_spin)
        solver_group.setLayout(solver_layout)

        # Кнопка запуска
        self.run_btn = QPushButton("Пересчитать")
        self.run_btn.clicked.connect(self.run_refinement)

        layout.addWidget(load_group)
        layout.addWidget(axes_group)
        layout.addWidget(smooth_group)
        layout.addWidget(solver_group)
        layout.addWidget(self.run_btn)
        layout.addStretch()
        panel.setLayout(layout)
        self.control_dock.setWidget(panel)

        # Подключаем кнопки загрузки
        self.load_data_btn.clicked.connect(self.load_data)
        self.load_machine_btn.clicked.connect(self.load_machine)

    def load_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Выберите файл результатов", "",
                                               "MAT files (*.mat)")
        if not fname:
            return
        try:
            mat = scipy.io.loadmat(fname)
            # Ожидаемые поля: 's', 'theta','Z','R','phi','tsn_pts','mandrel_pts','z_offset'
            self.s_array = mat['s'].flatten()
            self.tsn_pts = mat['tsn_pts']
            self.mandrel_pts = mat['mandrel_pts']
            self.z_offset = float(mat['z_offset'].flatten()[0])

            # Определяем оси из наличия полей (theta, Z, R, phi)
            possible_axes = ['theta', 'Z', 'R', 'phi']
            self.axes_names = []
            self.axes_units = []
            self.q_orig = []
            for ax in possible_axes:
                if ax in mat:
                    self.axes_names.append(ax)
                    unit = "рад" if ax == 'theta' or ax == 'phi' else "мм"
                    self.axes_units.append(unit)
                    self.q_orig.append(mat[ax].flatten())
            self.q_orig = np.column_stack(self.q_orig)  # (N, n_axes)

            # Обновить интерфейс: создать чекбоксы для осей
            self._update_axes_selector()
            # Построить сплайны для ТСН и оправки (нужны для пересчёта)
            self.tsn_traj = CubicSpline(self.s_array, self.tsn_pts, axis=0, bc_type='natural')
            self.mandrel_traj = CubicSpline(self.s_array, self.mandrel_pts, axis=0, bc_type='natural')
            self.d_tsn = self.tsn_traj.derivative(1)
            self.d_mandrel = self.mandrel_traj.derivative(1)

            # Отобразить исходные графики и таблицу
            self._update_plots(original=True)
            self._update_table()
            self._update_3d()

            QMessageBox.information(self, "Успех", f"Загружено {len(self.s_array)} точек, {len(self.axes_names)} осей")
            self.run_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {e}")

    def _update_axes_selector(self):
        # Очистить старые чекбоксы
        for i in reversed(range(self.axes_layout.count())):
            self.axes_layout.itemAt(i).widget().deleteLater()
        self.axis_checkboxes = []
        for idx, name in enumerate(self.axes_names):
            cb = QCheckBox(name)
            cb.setChecked(False)   # по умолчанию не фиксируем
            self.axes_layout.addWidget(cb)
            self.axis_checkboxes.append(cb)

    def load_machine(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Выберите параметры станка", "",
                                               "Pickle files (*.pkl)")
        if not fname:
            return
        try:
            with open(fname, 'rb') as f:
                params = pickle.load(f)
            # Ожидаем ring_radius, d_offset, возможно, type
            self.machine = Machine3AxisExact_ODE(ring_radius=params['ring_radius'],
                                                 d_offset=params['d_offset'])
            self.kin_model = KinematicModel(self.machine)
            QMessageBox.information(self, "Успех", "Модель станка загружена")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить станок: {e}")

    def _update_plots(self, original=True):
        # Очистить layout
        for i in reversed(range(self.plots_layout.count())):
            self.plots_layout.itemAt(i).widget().deleteLater()
        self.plot_canvases = []

        if self.s_array is None or self.q_orig is None:
            return

        for i, name in enumerate(self.axes_names):
            fig = Figure(figsize=(5, 3), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.plot(self.s_array, self.q_orig[:, i], 'b-', label='Исходная')
            if not original:
                # Если есть скорректированные данные, добавим позже
                pass
            ax.set_xlabel('s, мм')
            ax.set_ylabel(f'{name}, {self.axes_units[i]}')
            ax.set_title(name)
            ax.legend()
            ax.grid(True)
            self.plots_layout.addWidget(canvas)
            self.plot_canvases.append((canvas, i))
        # Добавим растяжку
        self.plots_layout.addStretch()

    def _update_table(self):
        if self.s_array is None or self.q_orig is None:
            return
        self.table_widget.setRowCount(len(self.s_array))
        self.table_widget.setColumnCount(len(self.axes_names) + 1)
        self.table_widget.setHorizontalHeaderLabels(['s, мм'] + self.axes_names)
        for i, s in enumerate(self.s_array):
            self.table_widget.setItem(i, 0, QTableWidgetItem(f"{s:.2f}"))
            for j, name in enumerate(self.axes_names):
                self.table_widget.setItem(i, j+1, QTableWidgetItem(f"{self.q_orig[i, j]:.4f}"))

    def _update_3d(self):
        if self.tsn_pts is None or self.mandrel_pts is None:
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=self.tsn_pts[:,0], y=self.tsn_pts[:,1], z=self.tsn_pts[:,2],
                                   mode='lines', name='ТСН (исходная)'))
        fig.add_trace(go.Scatter3d(x=self.mandrel_pts[:,0], y=self.mandrel_pts[:,1], z=self.mandrel_pts[:,2],
                                   mode='lines', name='Линия укладки'))
        fig.update_layout(scene=dict(aspectmode='data'))
        self.web_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def run_refinement(self):
        if self.kin_model is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите станок")
            return
        if self.s_array is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные")
            return

        # Собираем фиксированные оси
        fixed_indices = []
        fixed_funcs = []
        for idx, cb in enumerate(self.axis_checkboxes):
            if cb.isChecked():
                fixed_indices.append(idx)
                # Создаём функцию, возвращающую сглаженное значение
                s_param = self.smooth_param_slider.value()
                if s_param == 0:
                    spline = CubicSpline(self.s_array, self.q_orig[:, idx], bc_type='natural')
                else:
                    spline = UnivariateSpline(self.s_array, self.q_orig[:, idx], s=s_param)
                # Для использования в интеграторе нужна функция, принимающая s
                fixed_funcs.append(lambda s, spline=spline: spline(s))

        if not fixed_indices:
            QMessageBox.warning(self, "Предупреждение", "Не выбрано ни одной фиксируемой оси")
            return

        # Начальные значения для свободных осей (берём из первой точки)
        free_indices = [i for i in range(len(self.axes_names)) if i not in fixed_indices]
        q0_free = self.q_orig[0, free_indices].copy()

        # Параметры решателя
        method = 'adaptive' if self.solver_combo.currentIndex() == 0 else 'fixed_step'
        step = self.step_spin.value() if method == 'fixed_step' else None
        alpha = self.alpha_spin.value()

        # Запускаем поток
        self.compute_thread = ComputeThread(
            kin_model=self.kin_model,
            s_span=(self.s_array[0], self.s_array[-1]),
            q0_free=q0_free,
            fixed_funcs=fixed_funcs,
            fixed_indices=fixed_indices,
            tsn_func=self.tsn_traj,
            mandrel_func=self.mandrel_traj,
            d_tsn=self.d_tsn,
            d_mandrel=self.d_mandrel,
            s_eval=self.s_array,
            alpha=alpha,
            method=method,
            step=step
        )
        self.compute_thread.finished.connect(self.on_refinement_finished)
        self.compute_thread.error.connect(self.on_refinement_error)
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # индетерминированный прогресс
        self.compute_thread.start()

    def on_refinement_finished(self, result):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        coords_new = result['coords']   # (N, n_axes)
        # Обновить графики: добавить новые линии
        for canvas, idx in self.plot_canvases:
            fig = canvas.figure
            ax = fig.axes[0]
            # Удаляем старую линию "Скорректированная", если есть
            for line in ax.lines:
                if line.get_label() == 'Скорректированная':
                    line.remove()
            ax.plot(self.s_array, coords_new[:, idx], 'r--', label='Скорректированная')
            ax.legend()
            canvas.draw()

        # Обновить таблицу: добавить колонки скорректированных значений?
        # Для простоты покажем сообщение
        # Вычислим ошибку прямой задачи для проверки
        R_tsn_rec = np.zeros_like(self.tsn_pts)
        for i, s in enumerate(self.s_array):
            state = MachineState(coords_new[i])
            R_tsn_rec[i] = self.machine.forward(state)['point']
        target = np.array([self.tsn_traj(s) for s in self.s_array])
        error = np.linalg.norm(target - R_tsn_rec, axis=1)
        QMessageBox.information(self, "Результат",
                                f"Средняя ошибка прямой задачи: {np.mean(error):.3e} мм\n"
                                f"Максимальная ошибка: {np.max(error):.3e} мм")

        # Сохранение результата
        # (можно добавить кнопку, но для простоты сохраним автоматически)
        refined_data = {
            's': self.s_array,
            'coords': coords_new,
            'axes_names': self.axes_names,
            'tsn_pts': self.tsn_pts,
            'mandrel_pts': self.mandrel_pts,
            'z_offset': self.z_offset
        }
        scipy.io.savemat('refined_kinematics.mat', refined_data)
        self.status_bar.showMessage("Результаты сохранены в refined_kinematics.mat")

    def on_refinement_error(self, err_msg):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Ошибка расчёта", err_msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RefinementApp()
    window.show()
    sys.exit(app.exec_())