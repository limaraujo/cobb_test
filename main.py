from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem, QMenu
)
from PyQt6.QtGui import QPixmap, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QPointF, QTimer
import sys
import numpy as np

# ---------------- Funções de geometria ----------------
def ponto_interseccao(p1, p2, q1, q2):
    """Calcula o ponto de interseção de duas retas, ou None se paralelas."""
    x1, y1 = p1.x(), p1.y()
    x2, y2 = p2.x(), p2.y()
    x3, y3 = q1.x(), q1.y()
    x4, y4 = q2.x(), q2.y()

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:  # retas paralelas
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return QPointF(px, py)

def prolongar_reta_para_encontro(p1: QPointF, p2: QPointF, encontro: QPointF, largura: int, altura: int):
    """Prolonga a linha até os limites da imagem com margem de 5%."""
    p1_arr = np.array([p1.x(), p1.y()], dtype=float)
    p2_arr = np.array([p2.x(), p2.y()], dtype=float)
    dir_vec = p2_arr - p1_arr

    if np.linalg.norm(dir_vec) == 0:
        return p1, p2

    encontro_arr = np.array([encontro.x(), encontro.y()])
    if np.dot(encontro_arr - p1_arr, dir_vec) > 0:
        origem = p1_arr
        dir_final = dir_vec / np.linalg.norm(dir_vec)
    else:
        origem = p2_arr
        dir_final = -dir_vec / np.linalg.norm(dir_vec)

    xmin, xmax = 0.05 * largura, 0.95 * largura
    ymin, ymax = 0.05 * altura, 0.95 * altura

    ts = []
    if dir_final[0] != 0:
        ts.extend([(xmin - origem[0]) / dir_final[0], (xmax - origem[0]) / dir_final[0]])
    if dir_final[1] != 0:
        ts.extend([(ymin - origem[1]) / dir_final[1], (ymax - origem[1]) / dir_final[1]])

    ts_validos = [t for t in ts if t >= 0]
    t_max = min(ts_validos) if ts_validos else 0

    ponto_ext = origem + dir_final * t_max
    return QPointF(origem[0], origem[1]), QPointF(ponto_ext[0], ponto_ext[1])

# ---------------- CobbAngleItem ----------------
class CobbAngleItem:
    def __init__(self, line1: 'LineConnection', line2: 'LineConnection', scene: QGraphicsScene):
        self.line1 = line1
        self.line2 = line2
        self.scene = scene

        # Linhas prolongadas visuais
        self.ext_line1 = QGraphicsLineItem()
        self.ext_line2 = QGraphicsLineItem()
        pen_ext = QPen(Qt.GlobalColor.green, 1, Qt.PenStyle.DashLine)
        self.ext_line1.setPen(pen_ext)
        self.ext_line2.setPen(pen_ext)
        scene.addItem(self.ext_line1)
        scene.addItem(self.ext_line2)

        # Calcula ângulo e atualiza prolongamentos
        self.angle_rad, self.angle_deg = self.calculate_angle()

        # Texto do ângulo
        self.text_item = QGraphicsTextItem(f"{self.angle_deg:.1f}°")
        self.text_item.setDefaultTextColor(Qt.GlobalColor.darkRed)
        self.text_item.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        scene.addItem(self.text_item)
        self.update_text_position()

        # Vincular atualizações às linhas
        self.line1.angles.append(self)
        self.line2.angles.append(self)

    def calculate_angle(self):
        p1, p2 = self.line1.p1, self.line1.p2
        q1, q2 = self.line2.p1, self.line2.p2
        v1 = np.array([p2.pos().x() - p1.pos().x(), p2.pos().y() - p1.pos().y()])
        v2 = np.array([q2.pos().x() - q1.pos().x(), q2.pos().y() - q1.pos().y()])

        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_theta = dot / norm if norm != 0 else 0
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        largura = self.scene.viewer.pixmap_item.pixmap().width()
        altura = self.scene.viewer.pixmap_item.pixmap().height()

        intersec = ponto_interseccao(p1.pos(), p2.pos(), q1.pos(), q2.pos())

        if intersec is not None:
            start1, end1 = prolongar_reta_para_encontro(p1.pos(), p2.pos(), intersec, largura, altura)
            start2, end2 = prolongar_reta_para_encontro(q1.pos(), q2.pos(), intersec, largura, altura)
            self.ext_line1.setLine(start1.x(), start1.y(), end1.x(), end1.y())
            self.ext_line2.setLine(start2.x(), start2.y(), end2.x(), end2.y())
        else:
            self.ext_line1.setLine(p1.pos().x(), p1.pos().y(), p2.pos().x(), p2.pos().y())
            self.ext_line2.setLine(q1.pos().x(), q1.pos().y(), q2.pos().x(), q2.pos().y())

        return theta_rad, np.degrees(theta_rad)

    def update(self):
        self.angle_rad, self.angle_deg = self.calculate_angle()
        self.text_item.setPlainText(f"{self.angle_deg:.1f}°")
        self.update_text_position()

    def update_text_position(self):
        x = (self.line1.p1.pos().x() + self.line2.p1.pos().x()) / 2
        y = (self.line1.p1.pos().y() + self.line2.p1.pos().y()) / 2
        self.text_item.setPos(x + 5, y + 5)

# ---------------- DraggablePoint ----------------
class DraggablePoint(QGraphicsEllipseItem):
    def __init__(self, pos: QPointF, r=6):
        super().__init__(-r, -r, 2*r, 2*r)
        self.setBrush(QColor("red"))
        self.setFlags(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(1)
        self.setPos(pos)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange:
            print(f"Ponto movido para: {value.x():.2f}, {value.y():.2f}")
        return super().itemChange(change, value)

# ---------------- LineConnection ----------------
class LineConnection:
    def __init__(self, scene, p1, p2, color=Qt.GlobalColor.blue, width=2):
        self.scene = scene
        self.p1 = p1
        self.p2 = p2
        self.angles = []

        self.line = QGraphicsLineItem()
        self.line.setPen(QPen(color, width))
        self.line.setFlag(QGraphicsLineItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.line.setFlag(QGraphicsLineItem.GraphicsItemFlag.ItemIsFocusable, True)
        self.line.mousePressEvent = self.on_click_line
        scene.addItem(self.line)
        self.update_line()

        self.p1.itemChange = self.wrap_item_change(self.p1)
        self.p2.itemChange = self.wrap_item_change(self.p2)

    def update_line(self):
        if not self.scene.viewer.pixmap_item:
            return
        self.line.setLine(
            self.p1.pos().x(), self.p1.pos().y(),
            self.p2.pos().x(), self.p2.pos().y()
        )

    def wrap_item_change(self, point):
        old_item_change = point.itemChange
        def new_item_change(change, value):
            if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange:
                self.update_line()
                for angle in self.angles:
                    angle.update()
            return old_item_change(change, value)
        return new_item_change

    def on_click_line(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            angles_associados = [angle for angle in self.angles if angle in self.scene.viewer.cobb_angles]
            if angles_associados:
                menu = QMenu()
                menu.addAction("Remover Ângulo de Cobb", lambda: self.remove_cobb_angle())
                menu.exec(event.screenPos())
        super(QGraphicsLineItem, self.line).mousePressEvent(event)

    def remove_cobb_angle(self):
        for angle in self.angles[:]:
            if angle in self.scene.viewer.cobb_angles:
                # Remove itens gráficos
                self.scene.removeItem(angle.text_item)
                self.scene.removeItem(angle.ext_line1)
                self.scene.removeItem(angle.ext_line2)
                
                # Remove a linha azul original
                if angle.line1.line in self.scene.items():
                    self.scene.removeItem(angle.line1.line)
                if angle.line2.line in self.scene.items():
                    self.scene.removeItem(angle.line2.line)
                
                # Remove referências aos pontos
                for ponto in [angle.line1.p1, angle.line1.p2, angle.line2.p1, angle.line2.p2]:
                    if ponto in self.scene.viewer.points:
                        self.scene.viewer.points.remove(ponto)
                    self.scene.removeItem(ponto)
                
                # Remove referências das linhas
                if angle in angle.line1.angles:
                    angle.line1.angles.remove(angle)
                if angle in angle.line2.angles:
                    angle.line2.angles.remove(angle)

                # Remove da lista global de ângulos
                self.scene.viewer.cobb_angles.remove(angle)
                
                print("Ângulo de Cobb, linhas e pontos associados removidos.")


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_factor = 1.15
        self.zoom_min = 0.1
        self.zoom_max = 10
        self.current_zoom = 1.0
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:  # ctrl+scroll = zoom
            delta = event.angleDelta().y() / 120  # cada “passo” do mouse
            factor = self.zoom_factor ** delta
            new_zoom = self.current_zoom * factor
            if self.zoom_min <= new_zoom <= self.zoom_max:
                self.scale(factor, factor)
                self.current_zoom = new_zoom
            event.accept()
        else:
            super().wheelEvent(event)  # scroll normal para pan


# ---------------- CustomScene ----------------
class CustomScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = None
        self.line_points = []

    def addPoint(self, pos):
        if self.viewer.adding_angle and self.viewer.pixmap_item.contains(pos):
            point = DraggablePoint(pos)
            self.addItem(point)
            self.viewer.points.append(point)
            self.line_points.append(point)
            print(f"Ponto adicionado em: {pos.x():.2f}, {pos.y():.2f}")

            if len(self.line_points) % 2 == 0:
                self.addConnectionLine()

            if len(self.line_points) == 4:
                self.viewer.calculate_angle()
                self.viewer.adding_angle = False
                self.line_points = []
                print("Modo de adicionar ângulo de Cobb desativado")

            return point

    def addConnectionLine(self):
        if len(self.line_points) >= 2:
            p1, p2 = self.line_points[-2], self.line_points[-1]
            self.viewer.lines.append(LineConnection(self, p1, p2))
            print(f"Linha criada entre: ({p1.pos().x():.2f},{p1.pos().y():.2f}) -> "
                  f"({p2.pos().x():.2f},{p2.pos().y():.2f})")

    def mousePressEvent(self, event):
        pos = event.scenePos()
        if self.viewer and self.viewer.pixmap_item and self.viewer.adding_angle:
            if self.viewer.pixmap_item.contains(pos):
                self.addPoint(pos)
        super().mousePressEvent(event)

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizador de Cobb Angle")
        self.setGeometry(100, 100, 900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Botões
        self.button_open = QPushButton("Selecionar Imagem")
        self.button_open.clicked.connect(self.open_image)
        layout.addWidget(self.button_open)

        self.button_angle = QPushButton("Adicionar Ângulo de Cobb")
        self.button_angle.clicked.connect(self.enable_add_angle)
        layout.addWidget(self.button_angle)

        self.button_zoom_in = QPushButton("+")
        self.button_zoom_in.clicked.connect(self.zoom_in)
        layout.addWidget(self.button_zoom_in)

        self.button_zoom_out = QPushButton("-")
        self.button_zoom_out.clicked.connect(self.zoom_out)
        layout.addWidget(self.button_zoom_out)

        # QGraphicsView
        self.view = ZoomableGraphicsView()
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # ativa pan
        layout.addWidget(self.view)

        # Configuração do zoom centralizado
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.scene = CustomScene()
        self.scene.viewer = self
        self.view.setScene(self.scene)

        # Variáveis
        self.pixmap_item = None
        self.points = []
        self.lines = []
        self.cobb_angles = []
        self.adding_angle = False
        self.zoom_factor = 1.15
        self.zoom_max = 10
        self.zoom_min = 0.1
        self.current_zoom = 1.0

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecione a Imagem", "", "Imagens (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            pixmap = QPixmap(path)
            self.scene.clear()
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.points = []
            self.lines = []
            self.cobb_angles = []
            self.current_zoom = 1.0

    def enable_add_angle(self):
        if not self.pixmap_item:
            print("Selecione uma imagem primeiro.")
            return
        self.adding_angle = True
        print("Modo de adicionar ângulo de Cobb ativado. Clique na imagem para adicionar 4 pontos (2 linhas).")

    def calculate_angle(self):
        if len(self.lines) >= 2:
            line1, line2 = self.lines[-2], self.lines[-1]
            angle_item = CobbAngleItem(line1, line2, self.scene)
            self.cobb_angles.append(angle_item)
            print(f"Ângulo de Cobb: {angle_item.angle_deg:.2f}°")
        else:
            print("Erro: É necessário ter pelo menos 2 linhas.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.current_zoom = 1.0

    # --------- Zoom com botões ----------
    def zoom_in(self):
        self._apply_zoom(self.zoom_factor)

    def zoom_out(self):
        self._apply_zoom(1 / self.zoom_factor)

    # --------- Zoom com roda do mouse ----------
    def wheelEvent(self, event):
        if self.pixmap_item:
            delta = event.angleDelta().y()
            if delta > 0:
                self._apply_zoom(self.zoom_factor)
            else:
                self._apply_zoom(1 / self.zoom_factor)

    # --------- Função de zoom centralizada com limites ----------
    def _apply_zoom(self, factor):
        new_zoom = self.current_zoom * factor
        if self.zoom_min <= new_zoom <= self.zoom_max:
            self.view.scale(factor, factor)
            self.current_zoom = new_zoom

# ---------------- Execução ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())