from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem, QMenu, QLabel,
    QColorDialog, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QColor, QPen, QFont, QPainter, QIcon
from PyQt6.QtCore import Qt, QPointF, QSize, QRectF,  Qt, QPointF, QSize, QRectF
import sys
import numpy as np
import os

def resource_path(relative_path):
    """Retorna o caminho absoluto de arquivos, compatível com PyInstaller."""
    if hasattr(sys, "_MEIPASS"):
        # Quando rodando como exe, PyInstaller cria pasta temporária _MEIPASS
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

"""
App para visualização e cálculo do ângulo de Cobb em imagens radiográficas.
"""
# ---------------- Funções de geometria ----------------
def log(msg):
    pass  # Troque por print(msg) ou logging se quiser depurar
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
        origem = p2_arr
        dir_final = dir_vec / np.linalg.norm(dir_vec)
    else:
        origem = p1_arr
        dir_final = -dir_vec / np.linalg.norm(dir_vec)

    xmin, xmax = 0.1 * largura, 0.9 * largura
    ymin, ymax = 0.1 * altura, 0.9 * altura

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
    """Representa o ângulo de Cobb entre duas linhas."""
    def __init__(self, line1: 'LineConnection', line2: 'LineConnection', scene: QGraphicsScene, COLOR=Qt.GlobalColor.blue):
        self.line1 = line1
        self.line2 = line2
        self.scene = scene
        self.line_color = COLOR
        self.font_size = 26  # Tamanho padrão da fonte
        self.is_resizing = False
        self.resize_start_pos = None
        self.resize_start_size = None

        # Linhas prolongadas visuais
        self.ext_line1 = QGraphicsLineItem()
        self.ext_line2 = QGraphicsLineItem()
        pen_ext = QPen(COLOR, 4, Qt.PenStyle.CustomDashLine)
        pen_ext.setDashPattern([4, 3])
        self.ext_line1.setPen(pen_ext)
        self.ext_line2.setPen(pen_ext)
        scene.addItem(self.ext_line1)
        scene.addItem(self.ext_line2)

        # Calcula ângulo e atualiza prolongamentos
        self.angle_deg = self.calculate_angle()

        # Texto do ângulo
        self.text_item = QGraphicsTextItem(f"{self.angle_deg:.1f}°")
        self.text_item.setZValue(2)
        self.text_item.setFlags(
            QGraphicsLineItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsLineItem.GraphicsItemFlag.ItemIsFocusable |
            QGraphicsLineItem.GraphicsItemFlag.ItemIsMovable
        )
        self.text_item.setDefaultTextColor(COLOR)
        self.text_item.setFont(QFont("Arial", self.font_size, QFont.Weight.Bold))
        
        # Conecta eventos do mouse
        self.text_item.mousePressEvent = self.on_text_press
        self.text_item.mouseMoveEvent = self.on_text_move
        self.text_item.mouseReleaseEvent = self.on_text_release
        
        scene.addItem(self.text_item)
        self.update_text_position()

        # Vincular atualizações às linhas
        self.line1.angles.append(self)
        self.line2.angles.append(self)

    def on_text_press(self, event):
        """Manipula clique no texto - inicia redimensionamento ou menu."""
        if event.button() == Qt.MouseButton.RightButton:
            # Menu de contexto para seleção de tamanho
            menu = QMenu()
            
            # Opções de tamanho da fonte
            sizes = [16, 20, 24, 26, 30, 36, 42, 48, 56, 64]
            for size in sizes:
                action = menu.addAction(f"Tamanho {size}")
                action.triggered.connect(lambda checked, s=size: self.change_font_size(s))
                if size == self.font_size:
                    action.setCheckable(True)
                    action.setChecked(True)
            
            menu.exec(event.screenPos())
            
        elif event.button() == Qt.MouseButton.LeftButton:
            # Verifica se clicou próximo à borda para redimensionar
            item_rect = self.text_item.boundingRect()
            local_pos = event.pos()
            
            # Área de redimensionamento nas bordas (últimos 10 pixels)
            resize_margin = 10
            near_right = local_pos.x() > item_rect.width() - resize_margin
            near_bottom = local_pos.y() > item_rect.height() - resize_margin
            
            if near_right or near_bottom:
                # Inicia modo de redimensionamento
                self.is_resizing = True
                self.resize_start_pos = event.scenePos()
                self.resize_start_size = self.font_size
                self.text_item.setCursor(Qt.CursorShape.SizeFDiagCursor)
                print("Modo de redimensionamento ativado. Arraste para redimensionar.")
            else:
                # Movimento normal do texto
                super(QGraphicsTextItem, self.text_item).mousePressEvent(event)

    def on_text_move(self, event):
        """Manipula movimento do mouse sobre o texto."""
        if self.is_resizing and self.resize_start_pos:
            # Calcula mudança na posição
            current_pos = event.scenePos()
            delta_x = current_pos.x() - self.resize_start_pos.x()
            delta_y = current_pos.y() - self.resize_start_pos.y()
            
            # Usa a maior variação para determinar o novo tamanho
            delta = max(delta_x, delta_y)
            
            # Calcula novo tamanho (sensibilidade ajustável)
            sensitivity = 0.2  # Quanto menor, menos sensível
            new_size = max(12, min(72, self.resize_start_size + delta * sensitivity))
            
            # Atualiza tamanho em tempo real
            self.font_size = int(new_size)
            font = QFont("Arial", self.font_size, QFont.Weight.Bold)
            self.text_item.setFont(font)
            
        else:
            # Verifica se está sobre área de redimensionamento para mudar cursor
            item_rect = self.text_item.boundingRect()
            local_pos = event.pos()
            
            resize_margin = 10
            near_right = local_pos.x() > item_rect.width() - resize_margin
            near_bottom = local_pos.y() > item_rect.height() - resize_margin
            
            if near_right or near_bottom:
                self.text_item.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.text_item.setCursor(Qt.CursorShape.ArrowCursor)
                # Movimento normal
                super(QGraphicsTextItem, self.text_item).mouseMoveEvent(event)

    def on_text_release(self, event):
        """Manipula soltar o botão do mouse."""
        if self.is_resizing:
            self.is_resizing = False
            self.resize_start_pos = None
            self.resize_start_size = None
            self.text_item.setCursor(Qt.CursorShape.ArrowCursor)
            print(f"Redimensionamento concluído. Novo tamanho: {self.font_size}")
        else:
            super(QGraphicsTextItem, self.text_item).mouseReleaseEvent(event)

    def change_font_size(self, new_size):
        """Altera o tamanho da fonte do texto."""
        self.font_size = new_size
        font = QFont("Arial", self.font_size, QFont.Weight.Bold)
        self.text_item.setFont(font)
        print(f"Tamanho da fonte alterado para: {new_size}")

    def calculate_angle(self):        
        p1, p2 = self.line1.p1, self.line1.p2
        q1, q2 = self.line2.p1, self.line2.p2

        if p1.pos().x() > p2.pos().x():
            p1, p2 = p2, p1
        if q1.pos().x() > q2.pos().x():
            q1, q2 = q2, q1

        # Vetores normalizados
        v1 = np.array([p2.pos().x() - p1.pos().x(), p2.pos().y() - p1.pos().y()])
        v2 = np.array([q2.pos().x() - q1.pos().x(), q2.pos().y() - q1.pos().y()])

        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_theta = dot / norm if norm != 0 else 0
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        theta_deg = np.degrees(theta_rad)

        # Prolonga linhas para visualização
        largura = self.scene.viewer.pixmap_item.pixmap().width()
        altura = self.scene.viewer.pixmap_item.pixmap().height()
        intersec = ponto_interseccao(p1.pos(), p2.pos(), q1.pos(), q2.pos())
        for ext_line, a, b in [
            (self.ext_line1, p1.pos(), p2.pos()),
            (self.ext_line2, q1.pos(), q2.pos())
        ]:
            start, end = prolongar_reta_para_encontro(a, b, intersec, largura, altura) if intersec is not None else (a, b)
            ext_line.setLine(start.x(), start.y(), end.x(), end.y())

        return theta_deg

    def update(self):
        self.angle_deg = self.calculate_angle()
        self.text_item.setPlainText(f"{self.angle_deg:.1f}°")
        # Mantém o tamanho da fonte atual
        font = QFont("Arial", self.font_size, QFont.Weight.Bold)
        self.text_item.setFont(font)
        self.update_text_position()

    def update_text_position(self):
        # Posição do texto no centro aproximado das linhas
        x = (self.line1.p1.pos().x() + self.line2.p1.pos().x()) / 2
        y = (self.line1.p1.pos().y() + self.line2.p1.pos().y()) / 2
        self.text_item.setPos(x + 5, y + 5)


# ---------------- DraggablePoint ----------------
class DraggablePoint(QGraphicsEllipseItem):
    """Ponto arrastável na cena gráfica."""
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
            pass
        return super().itemChange(change, value)

# ---------------- LineConnection ----------------
class LineConnection:
    """Conexão entre dois pontos, representando uma linha manipulável."""
    def __init__(self, scene, p1, p2, color=Qt.GlobalColor.blue, width=4):
        self.scene = scene
        self.p1 = p1
        self.p2 = p2
        self.angles = []

        self.line = QGraphicsLineItem()
        self.line.setPen(QPen(color, width))
        self.line.setFlags(
            QGraphicsLineItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsLineItem.GraphicsItemFlag.ItemIsFocusable
        )
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
        
        for angle in self.angles:
            angle.update()

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
        if event.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton]:
            menu = QMenu()
            if any(angle in self.scene.viewer.cobb_angles for angle in self.angles):
                menu.addAction("Remover Ângulo de Cobb").triggered.connect(self.remove_cobb_angle)
            menu.addAction("Mudar Cor da Linha").triggered.connect(self.change_line_color)
            menu.exec(event.screenPos())
        super(QGraphicsLineItem, self.line).mousePressEvent(event)

    def change_line_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            for angle in self.angles:
                if angle in self.scene.viewer.cobb_angles:
                    angle.line1.line.setPen(QPen(color, 4))
                    angle.line2.line.setPen(QPen(color, 4))
                    
                    pen_ext = QPen(color, 4, Qt.PenStyle.CustomDashLine)
                    pen_ext.setDashPattern([4, 3])
                    angle.ext_line1.setPen(pen_ext)
                    angle.ext_line2.setPen(pen_ext)
                    
                    angle.text_item.setDefaultTextColor(color)
                    
                    angle.line_color = color

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
    """QGraphicsView com suporte a zoom centralizado e limites."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_factor = 1.1
        self.zoom_min = 0.5
        self.zoom_max = 5
        self.current_zoom = 1.0
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:  # ctrl+scroll = zoom
            delta = event.angleDelta().y() / 120
            self.apply_zoom(self.zoom_factor ** delta)
            event.accept()
        else:
            super().wheelEvent(event)
    
    # --------- Função de zoom centralizada com limites ----------
    def apply_zoom(self, factor):
        """Aplica o zoom com limites definidos."""
        new_zoom = self.current_zoom * factor
        if new_zoom > self.zoom_max:
            factor = self.zoom_max / self.current_zoom
            new_zoom = self.zoom_max
        elif new_zoom < self.zoom_min:
            factor = self.zoom_min / self.current_zoom
            new_zoom = self.zoom_min
        self.scale(factor, factor)
        self.current_zoom = new_zoom

# ---------------- CustomScene ----------------
class CustomScene(QGraphicsScene):
    """Cena customizada para manipulação dos pontos e linhas do Cobb."""
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

            if len(self.line_points) % 2 == 0 :
                self.addConnectionLine()

            if len(self.line_points) == 4:
                self.viewer.calculate_angle()
                self.viewer.adding_angle = False
                self.viewer.cobb_button.setStyleSheet(self.viewer.button_style)
                self.line_points = []
                print("Modo de adicionar ângulo de Cobb desativado")

            return point

    def addConnectionLine(self):
        if len(self.line_points) >= 2:
            p1, p2 = self.line_points[-2], self.line_points[-1]
            self.viewer.lines.append(LineConnection(self, p1, p2, color=self.viewer.selected_cobb_color))
            print(f"Linha criada entre: ({p1.pos().x():.2f},{p1.pos().y():.2f}) -> "
                  f"({p2.pos().x():.2f},{p2.pos().y():.2f})")

    def mousePressEvent(self, event):
        pos = event.scenePos()
        if self.viewer and self.viewer.pixmap_item and self.viewer.adding_angle:
            if self.viewer.pixmap_item.contains(pos):
                min_distance = 6
                too_close = any(
                    ((pos.x() - p.pos().x())**2 + (pos.y() - p.pos().y())**2)**0.5 < min_distance 
                    for p in self.viewer.points
                )
                if not too_close:
                    self.addPoint(pos)
        super().mousePressEvent(event)

class ImageViewer(QMainWindow):
    """Janela principal do aplicativo de visualização do ângulo de Cobb."""
    button_style = """
    QPushButton {
        background-color: transparent;   /* sem cor de fundo */
        color: black;                     /* texto preto */
        border: 0.5px solid rgba(128, 128, 128, 0.2);       /* borda verde */
        border-radius: 5px;               /* cantos arredondados */
        padding: 8px 16px;                /* espaço interno */
        font-size: 14px;                  /* tamanho da fonte */
        min-height: 26px;  
    }

    QPushButton:hover {
        background-color: #e0e0e0;
        border: 1px solid #888;
    }
    """
        
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizador de Cobb Angle")
        self.setGeometry(100, 100, 900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.selected_cobb_color = Qt.GlobalColor.blue  

        buttons = [
            ("Selecionar Arquivo", "icons/upload.png", self.open_image),
            ("Salvar Imagem", "icons/download.png", self.save_image),
            ("Ângulo de Cobb", "icons/angle.png", self.enable_add_angle),
            ("", "icons/zoom_in.png", self.zoom_in),
            ("Reset", "icons/zoom_reset.png", self.reset_zoom),
            ("", "icons/zoom_out.png", self.zoom_out),
        ]

        buttons_layout = QHBoxLayout()
        for text, icon_path, slot in buttons:
            btn = QPushButton(text)
            # usa resource_path para o caminho do ícone
            btn.setIcon(QIcon(resource_path(icon_path)))
            btn.setIconSize(QSize(24, 24))
            btn.setStyleSheet(self.button_style)
            btn.clicked.connect(slot)
            buttons_layout.addWidget(btn)
            if text == "Ângulo de Cobb":
                self.cobb_button = btn

            
        layout.addLayout(buttons_layout)
        
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

        self.footer = QLabel("©2025 limaraujo.")
        self.footer.setStyleSheet("background-color: transparent; padding: 5px;")
        self.footer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.footer)  # adicione direto, sem addStretch()

        # Variáveis
        self.pixmap_item = None
        self.points = []
        self.lines = []
        self.cobb_angles = []
        self.adding_angle = False


    def open_image(self):
    # Abre uma imagem e reseta a cena.
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecione a Imagem", "", "Imagens (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            pixmap = QPixmap(path)
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            self.view.resetTransform()
            self.view.current_zoom = 1.0
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.points = []
            self.lines = []
            self.cobb_angles = []

    def open_color_dialog(self):
        if not self.pixmap_item:
            print("Selecione uma imagem primeiro.")
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_cobb_color = color
        else:
            self.selected_cobb_color = Qt.GlobalColor.blue
        self.enable_add_angle()

    def enable_add_angle(self):
        self.selected_cobb_color = Qt.GlobalColor.blue
        self.adding_angle = True
        if self.cobb_button:
            self.cobb_button.setStyleSheet(
                self.button_style +
                "QPushButton { background-color: #1976d2; border: 1px solid rgba(10, 73, 112, 1); background-color: rgba(52, 139, 210, 1); }"
            )
        print("Modo de adicionar ângulo de Cobb ativado. Clique na imagem para adicionar 4 pontos (2 linhas).")

    def calculate_angle(self):
        if len(self.lines) >= 2:
            line1, line2 = self.lines[-2], self.lines[-1]
            # Usa a cor selecionada
            angle_item = CobbAngleItem(line1, line2, self.scene, COLOR=self.selected_cobb_color)
            self.cobb_angles.append(angle_item)
            print(f"Ângulo de Cobb: {angle_item.angle_deg:.2f}°")
        else:
            print("Erro: É necessário ter pelo menos 2 linhas.")
            
    def save_image(self):
        if not self.pixmap_item:
            print("Nenhuma imagem carregada para salvar.")
            return

        rect = self.scene.itemsBoundingRect()
        image = QPixmap(int(rect.width()), int(rect.height()))
        image.fill(Qt.GlobalColor.white)

        painter = QPainter(image)
        self.scene.render(painter, target=QRectF(image.rect()), source=rect)
        painter.end()

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Salvar Imagem", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if path:
            ext = os.path.splitext(path)[1]
            if not ext:
                ext_map = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp"}
            for key, val in ext_map.items():
                if key in selected_filter:
                    path += val
                break
            print("Imagem salva." if image.save(path) else "Erro ao salvar imagem.")
                
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.view.resetTransform()
            self.view.current_zoom = 1.0
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        # --------- Zoom com botões ----------
    def zoom_in(self):
        self.view.apply_zoom(self.view.zoom_factor)

    def zoom_out(self):
        self.view.apply_zoom(1 / self.view.zoom_factor)

    def reset_zoom(self):
        self.view.resetTransform()
        self.view.current_zoom = 1.0
        if self.pixmap_item:
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            
    def open_settings_menu(self):
        print("menu aberto")
        pass


# ---------------- Execução ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())