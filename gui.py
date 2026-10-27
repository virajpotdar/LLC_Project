"""Tkinter user interface for MatrixMaster."""
from __future__ import annotations

import csv
import json
import math
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .core import MatrixOperationError, as_matrix, calculate


APP_TITLE = "MatrixMaster | Matrix Calculator"
MAX_DIMENSION = 8


class MatrixEditor(ttk.LabelFrame):
    """Editable matrix grid with dimension and file helpers."""

    def __init__(self, master, title: str, default_values, accent: str):
        super().__init__(master, text=title, padding=10)
        self.accent = accent
        self.rows_var = tk.IntVar(value=len(default_values))
        self.cols_var = tk.IntVar(value=len(default_values[0]))
        self.entries: list[list[ttk.Entry]] = []
        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(fill="both", expand=True, pady=(6, 8))

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text="Rows").pack(side="left")
        ttk.Spinbox(controls, from_=1, to=MAX_DIMENSION, width=3, textvariable=self.rows_var).pack(side="left", padx=(4, 10))
        ttk.Label(controls, text="Columns").pack(side="left")
        ttk.Spinbox(controls, from_=1, to=MAX_DIMENSION, width=3, textvariable=self.cols_var).pack(side="left", padx=(4, 8))
        ttk.Button(controls, text="Apply size", command=self.apply_size).pack(side="left")

        tools = ttk.Frame(self)
        tools.pack(fill="x", pady=(8, 0))
        ttk.Button(tools, text="Random", command=self.random_fill).pack(side="left")
        ttk.Button(tools, text="Identity", command=self.identity_fill).pack(side="left", padx=5)
        ttk.Button(tools, text="Clear", command=self.clear).pack(side="left")
        ttk.Button(tools, text="Load CSV", command=self.load_csv).pack(side="right")
        self.set_values(default_values)

    def apply_size(self):
        try:
            rows, cols = int(self.rows_var.get()), int(self.cols_var.get())
            if not (1 <= rows <= MAX_DIMENSION and 1 <= cols <= MAX_DIMENSION):
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showwarning("Invalid dimensions", f"Choose rows and columns from 1 to {MAX_DIMENSION}.")
            return
        self.set_values(np.zeros((rows, cols), dtype=float))

    def set_values(self, values):
        matrix = as_matrix(values)
        if matrix.shape[0] > MAX_DIMENSION or matrix.shape[1] > MAX_DIMENSION:
            raise MatrixOperationError(f"Matrix size is limited to {MAX_DIMENSION}×{MAX_DIMENSION} in this mini-project.")
        self.rows_var.set(matrix.shape[0])
        self.cols_var.set(matrix.shape[1])
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.entries = []
        for r in range(matrix.shape[0]):
            row_entries = []
            for c in range(matrix.shape[1]):
                entry = ttk.Entry(self.grid_frame, width=7, justify="center")
                entry.grid(row=r, column=c, padx=3, pady=3, ipady=4, sticky="nsew")
                entry.insert(0, _format_number(matrix[r, c]))
                row_entries.append(entry)
            self.entries.append(row_entries)
        for c in range(matrix.shape[1]):
            self.grid_frame.columnconfigure(c, weight=1)

    def get_values(self):
        values = []
        for r, row in enumerate(self.entries, start=1):
            current = []
            for c, entry in enumerate(row, start=1):
                raw = entry.get().strip()
                try:
                    number = float(raw)
                except ValueError as exc:
                    raise MatrixOperationError(f"{self['text']} cell at row {r}, column {c} is not a valid number.") from exc
                if not math.isfinite(number):
                    raise MatrixOperationError("Please enter finite numeric values only.")
                current.append(number)
            values.append(current)
        return as_matrix(values, self["text"])

    def random_fill(self):
        self.set_values(np.random.randint(-9, 10, size=(self.rows_var.get(), self.cols_var.get())))

    def identity_fill(self):
        rows, cols = self.rows_var.get(), self.cols_var.get()
        self.set_values(np.eye(rows, cols))

    def clear(self):
        self.set_values(np.zeros((self.rows_var.get(), self.cols_var.get())))

    def load_csv(self):
        path = filedialog.askopenfilename(title="Load matrix from CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            matrix = np.loadtxt(path, delimiter=",", ndmin=2)
            self.set_values(matrix)
        except Exception as exc:
            messagebox.showerror("Could not load CSV", str(exc))


class MatrixMasterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1180x820")
        self.root.minsize(980, 700)
        self.root.configure(bg="#eef2f7")
        self.history: list[dict] = []
        self.last_result = None
        self.last_title = "Result"
        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabel", background="#eef2f7", foreground="#1e293b", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground="#ffffff", background="#14213d")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#dbeafe", background="#14213d")
        style.configure("TLabelframe", background="#ffffff", bordercolor="#d8e0eb")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#14213d", font=("Segoe UI", 11, "bold"))
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=(8, 6))
        style.configure("Accent.TButton", background="#2563eb", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#1d4ed8")])
        style.configure("Treeview", rowheight=25, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#14213d", padx=22, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="MatrixMaster", bg="#14213d", fg="white", font=("Segoe UI", 23, "bold")).pack(anchor="w")
        tk.Label(header, text="A simple workspace for matrix calculations and visual analysis", bg="#14213d", fg="#cbd5e1", font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 0))

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        editors = ttk.Frame(outer)
        editors.pack(fill="x")
        editors.columnconfigure(0, weight=1)
        editors.columnconfigure(1, weight=1)
        self.matrix_a = MatrixEditor(editors, "MATRIX A", [[1, 2], [3, 4]], "#2563eb")
        self.matrix_a.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self.matrix_b = MatrixEditor(editors, "MATRIX B", [[5, 6], [7, 8]], "#0f766e")
        self.matrix_b.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        middle = ttk.Frame(outer)
        middle.pack(fill="x", pady=12)
        operations = ttk.LabelFrame(middle, text="OPERATIONS", padding=10)
        operations.pack(side="left", fill="both", expand=True)
        buttons = [
            ("Add (A + B)", "Add"), ("Subtract (A - B)", "Subtract"),
            ("Multiply (A × B)", "Multiply"), ("Element-wise (A ⊙ B)", "Element-wise"),
            ("Transpose A", "Transpose A"), ("Transpose B", "Transpose B"),
            ("Determinant A", "Determinant A"), ("Determinant B", "Determinant B"),
            ("Inverse A", "Inverse A"), ("Inverse B", "Inverse B"),
            ("Rank A", "Rank A"), ("Rank B", "Rank B"),
            ("Trace A", "Trace A"), ("Trace B", "Trace B"),
        ]
        for i, (operation, label) in enumerate(buttons):
            ttk.Button(operations, text=label, command=lambda op=operation: self.run_operation(op)).grid(row=i // 7, column=i % 7, padx=3, pady=4, sticky="ew")
        for c in range(7):
            operations.columnconfigure(c, weight=1)

        scalar_frame = ttk.Frame(operations)
        scalar_frame.grid(row=2, column=0, columnspan=7, sticky="w", pady=(5, 0))
        ttk.Label(scalar_frame, text="Scalar value:").pack(side="left")
        self.scalar_var = tk.StringVar(value="2")
        ttk.Entry(scalar_frame, textvariable=self.scalar_var, width=9).pack(side="left", padx=7)
        ttk.Button(scalar_frame, text="Multiply A by scalar", command=lambda: self.run_operation("Scalar × A")).pack(side="left")
        ttk.Button(scalar_frame, text="Swap A and B", command=self.swap_matrices).pack(side="right")

        result_history = ttk.Frame(outer)
        result_history.pack(fill="both", expand=True)
        result_history.columnconfigure(0, weight=3)
        result_history.columnconfigure(1, weight=2)
        result_history.rowconfigure(0, weight=1)
        result_box = ttk.LabelFrame(result_history, text="RESULT & VISUALIZATION", padding=10)
        result_box.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        result_box.rowconfigure(1, weight=1)
        result_box.columnconfigure(0, weight=1)
        result_toolbar = ttk.Frame(result_box)
        result_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self.result_title_var = tk.StringVar(value="Run an operation to see the result")
        ttk.Label(result_toolbar, textvariable=self.result_title_var, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(result_toolbar, text="Heatmap", command=self.show_heatmap).pack(side="right", padx=(5, 0))
        ttk.Button(result_toolbar, text="Export CSV", command=self.export_result).pack(side="right", padx=5)
        ttk.Button(result_toolbar, text="Copy", command=self.copy_result).pack(side="right")

        result_content = ttk.Frame(result_box)
        result_content.grid(row=1, column=0, sticky="nsew")
        result_content.columnconfigure(0, weight=1)
        result_content.columnconfigure(1, weight=1)
        result_content.rowconfigure(0, weight=1)
        self.result_text = tk.Text(result_content, height=10, width=36, font=("Consolas", 11), bg="#f8fafc", fg="#0f172a", relief="flat", padx=12, pady=10, wrap="none")
        self.result_text.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.result_text.insert("1.0", "Your answer will appear here.\n\nTip: Try Add, Multiply, or Transpose.")
        self.result_text.configure(state="disabled")
        self.figure = Figure(figsize=(4.2, 2.8), dpi=90, facecolor="#ffffff")
        self.axes = self.figure.add_subplot(111)
        self.colorbar = None
        self.axes.set_title("Matrix heatmap")
        self.axes.text(0.5, 0.5, "Run a matrix operation\nthen click Heatmap", ha="center", va="center", transform=self.axes.transAxes, color="#64748b")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=result_content)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        history_box = ttk.LabelFrame(result_history, text="OPERATION HISTORY", padding=10)
        history_box.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        history_box.rowconfigure(0, weight=1)
        history_box.columnconfigure(0, weight=1)
        self.history_tree = ttk.Treeview(history_box, columns=("time", "operation", "size"), show="headings", height=9)
        self.history_tree.heading("time", text="Time")
        self.history_tree.heading("operation", text="Operation")
        self.history_tree.heading("size", text="Output")
        self.history_tree.column("time", width=62, anchor="center")
        self.history_tree.column("operation", width=145)
        self.history_tree.column("size", width=58, anchor="center")
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(history_box, orient="vertical", command=self.history_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.history_tree.configure(yscrollcommand=scroll.set)
        ttk.Button(history_box, text="Clear history", command=self.clear_history).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.status_var = tk.StringVar(value="Ready • Enter values in Matrix A and Matrix B")
        status = tk.Label(self.root, textvariable=self.status_var, bg="#dbeafe", fg="#1e3a8a", anchor="w", padx=15, pady=7, font=("Segoe UI", 9))
        status.pack(fill="x", side="bottom")

    def run_operation(self, operation: str):
        try:
            a = self.matrix_a.get_values()
            b = self.matrix_b.get_values()
            scalar = None
            if operation == "Scalar × A":
                try:
                    scalar = float(self.scalar_var.get())
                except ValueError as exc:
                    raise MatrixOperationError("Enter a valid number for the scalar value.") from exc
                if not math.isfinite(scalar):
                    raise MatrixOperationError("Scalar must be a finite number.")
            result = calculate(operation, a, b, scalar)
            self.last_result = result
            self.last_title = operation
            self.result_title_var.set(operation)
            self._display_result(result)
            self._add_history(operation, result)
            self.status_var.set(f"Completed: {operation}")
        except MatrixOperationError as exc:
            messagebox.showwarning("Cannot perform operation", str(exc))
            self.status_var.set("Operation needs attention")
        except Exception as exc:
            messagebox.showerror("Unexpected error", str(exc))
            self.status_var.set("An unexpected error occurred")

    def _display_result(self, result):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        if np.isscalar(result):
            self.result_text.insert("1.0", f"{_format_number(result)}\n")
        else:
            self.result_text.insert("1.0", _matrix_to_text(result))
        self.result_text.configure(state="disabled")

    def _add_history(self, operation, result):
        if isinstance(result, np.ndarray):
            size = f"{result.shape[0]}×{result.shape[1]}"
            value = result.tolist()
        else:
            size = "scalar"
            value = float(result)
        item = {"time": datetime.now().strftime("%H:%M:%S"), "operation": operation, "size": size, "result": value}
        self.history.insert(0, item)
        self.history_tree.insert("", 0, values=(item["time"], operation, size))
        self.history_tree.yview_moveto(0)

    def show_heatmap(self):
        if self.last_result is None:
            messagebox.showinfo("No result", "Run a matrix operation first.")
            return
        if not isinstance(self.last_result, np.ndarray):
            messagebox.showinfo("Heatmap unavailable", "Heatmaps are available for matrix results, not scalar results.")
            return
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.axes.clear()
        image = self.axes.imshow(self.last_result, cmap="YlGnBu", aspect="auto")
        self.axes.set_title(self.last_title, fontsize=10, pad=10)
        self.axes.set_xlabel("Column")
        self.axes.set_ylabel("Row")
        self.axes.set_xticks(range(self.last_result.shape[1]))
        self.axes.set_yticks(range(self.last_result.shape[0]))
        for (r, c), value in np.ndenumerate(self.last_result):
            self.axes.text(c, r, _format_number(value), ha="center", va="center", color="#172554", fontsize=8)
        self.colorbar = self.figure.colorbar(image, ax=self.axes, fraction=0.046, pad=0.04)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set("Heatmap updated")

    def export_result(self):
        if self.last_result is None:
            messagebox.showinfo("No result", "Run an operation before exporting.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="matrix_result.csv")
        if not path:
            return
        try:
            if np.isscalar(self.last_result):
                np.savetxt(path, np.array([[self.last_result]]), delimiter=",", fmt="%.8g")
            else:
                np.savetxt(path, self.last_result, delimiter=",", fmt="%.8g")
            self.status_var.set(f"Result exported: {Path(path).name}")
            messagebox.showinfo("Export complete", "Your result was saved as a CSV file.")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))

    def copy_result(self):
        if self.last_result is None:
            messagebox.showinfo("No result", "Run an operation before copying.")
            return
        text = str(self.last_result.tolist() if isinstance(self.last_result, np.ndarray) else self.last_result)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Result copied to clipboard")

    def swap_matrices(self):
        try:
            a = self.matrix_a.get_values()
            b = self.matrix_b.get_values()
            self.matrix_a.set_values(b)
            self.matrix_b.set_values(a)
            self.status_var.set("Matrix A and Matrix B swapped")
        except MatrixOperationError as exc:
            messagebox.showwarning("Cannot swap matrices", str(exc))

    def clear_history(self):
        self.history.clear()
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        self.status_var.set("Operation history cleared")


def _format_number(value) -> str:
    value = float(value)
    if abs(value) < 1e-10:
        value = 0.0
    return f"{value:.4f}".rstrip("0").rstrip(".") if not value.is_integer() else str(int(value))


def _matrix_to_text(matrix: np.ndarray) -> str:
    rows = [[_format_number(value) for value in row] for row in matrix]
    widths = [max(len(row[c]) for row in rows) for c in range(matrix.shape[1])]
    return "\n".join("[ " + "  ".join(value.rjust(widths[c]) for c, value in enumerate(row)) + " ]" for row in rows)


def main():
    root = tk.Tk()
    MatrixMasterApp(root)
    root.mainloop()
