"""Application theme: one calm light palette for the whole studio."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QPushButton

ACCENT = "#0f766e"
ACCENT_DARK = "#115e59"
SURFACE = "#ffffff"
BACKGROUND = "#f3f5f7"
BORDER = "#e2e8f0"
TEXT = "#1f2937"
MUTED = "#64748b"

STYLESHEET = """
QWidget {
    color: #1f2937;
    font-size: 13px;
}
QMainWindow, QDialog {
    background: #f3f5f7;
}
QToolBar {
    background: #ffffff;
    border: none;
    border-bottom: 1px solid #e2e8f0;
    padding: 8px 14px;
    spacing: 6px;
}
QToolBar QToolButton {
    color: #334155;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 7px 12px;
    font-weight: 600;
}
QToolBar QToolButton:hover {
    background: #f1f5f9;
    border-color: #e2e8f0;
}
QToolBar QToolButton:pressed {
    background: #e2e8f0;
}
QToolBar::separator {
    background: #e2e8f0;
    width: 1px;
    margin: 6px 8px;
}
QLabel#BrandDot {
    color: #0f766e;
    font-size: 18px;
    padding: 0 0 0 2px;
}
QLabel#Brand {
    color: #0f172a;
    font-size: 17px;
    font-weight: 700;
    padding-right: 4px;
}
QLabel#BrandSub {
    color: #94a3b8;
    font-size: 12px;
    padding-right: 8px;
}
QLabel#PathChip {
    color: #475569;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 5px 10px;
}
QLabel#DeviceChip {
    color: #64748b;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 9px;
    padding: 2px 8px;
}
QLabel#Muted {
    color: #64748b;
}
QLabel#Prompt {
    color: #0f172a;
    font-weight: 600;
}
QLabel#Warning {
    color: #b45309;
}
QLabel#DialogTitle {
    color: #0f172a;
    font-size: 18px;
    font-weight: 700;
}
QLabel#DialogSubtitle {
    color: #64748b;
}
QTabWidget::pane {
    border: 0;
    background: transparent;
}
QTabBar::tab {
    background: transparent;
    color: #64748b;
    padding: 10px 18px;
    margin-right: 2px;
    border: 0;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
QTabBar::tab:hover {
    color: #0f172a;
    background: #eef2f7;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}
QTabBar::tab:selected {
    color: #0f766e;
    border-bottom: 2px solid #0f766e;
    background: transparent;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 14px 14px 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 5px;
    color: #334155;
    background: #ffffff;
    font-weight: 600;
}
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #0f766e;
    selection-color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border-color: #0f766e;
}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
    background: #f8fafc;
    color: #94a3b8;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    selection-background-color: #ccfbf1;
    selection-color: #134e4a;
    outline: 0;
}
QPushButton {
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #f8fafc;
    border-color: #94a3b8;
}
QPushButton:pressed {
    background: #e2e8f0;
}
QPushButton:disabled {
    color: #94a3b8;
    background: #f1f5f9;
    border-color: #e2e8f0;
}
QPushButton[primary="true"] {
    background: #0f766e;
    color: #ffffff;
    border-color: #0f766e;
}
QPushButton[primary="true"]:hover {
    background: #115e59;
    border-color: #115e59;
}
QPushButton[primary="true"]:pressed {
    background: #134e4a;
}
QPushButton[primary="true"]:disabled {
    background: #99f6e4;
    border-color: #99f6e4;
    color: #ecfeff;
}
QPushButton[accent="true"] {
    color: #0f766e;
    border-color: #5eead4;
    background: #f0fdfa;
}
QPushButton[accent="true"]:hover {
    background: #ccfbf1;
    border-color: #0f766e;
}
QPushButton[accent="true"]:disabled {
    color: #94a3b8;
    border-color: #e2e8f0;
    background: #f8fafc;
}
QPushButton[danger="true"] {
    color: #b91c1c;
    border-color: #fecaca;
    background: #fef2f2;
}
QPushButton[danger="true"]:hover {
    background: #fee2e2;
    border-color: #f87171;
}
QPushButton[danger="true"]:disabled {
    color: #94a3b8;
    border-color: #e2e8f0;
    background: #f8fafc;
}
QPushButton:default {
    background: #0f766e;
    color: #ffffff;
    border-color: #0f766e;
}
QPushButton:default:hover {
    background: #115e59;
    border-color: #115e59;
}
QListWidget {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 4px;
    outline: 0;
}
QListWidget::item {
    padding: 6px 8px;
}
QListWidget::item:selected {
    background: #ccfbf1;
    color: #134e4a;
}
QListWidget::item:hover {
    background: #f1f5f9;
}
QProgressBar {
    background: #e2e8f0;
    border: none;
    border-radius: 6px;
    height: 16px;
    text-align: center;
    color: #334155;
    font-size: 11px;
}
QProgressBar::chunk {
    background: #0f766e;
    border-radius: 6px;
}
QCheckBox {
    spacing: 7px;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #cbd5e1;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #94a3b8;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
}
QStatusBar::item {
    border: 0;
}
QPlainTextEdit#LogPane {
    background: #0f172a;
    color: #d1fae5;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 8px;
    selection-background-color: #0f766e;
    selection-color: #ffffff;
}
QToolTip {
    background: #0f172a;
    color: #f8fafc;
    border: none;
    padding: 5px 8px;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
"""


def _role(button: QPushButton, name: str) -> QPushButton:
    button.setProperty(name, True)
    button.style().unpolish(button)
    button.style().polish(button)
    return button


def primary(button: QPushButton) -> QPushButton:
    """The main action on a page or in a group."""
    return _role(button, "primary")


def accent(button: QPushButton) -> QPushButton:
    """A secondary but important action."""
    return _role(button, "accent")


def danger(button: QPushButton) -> QPushButton:
    """A destructive action."""
    return _role(button, "danger")


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#94a3b8"))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
