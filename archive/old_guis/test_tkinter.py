#!/usr/bin/env python3

import tkinter as tk
from tkinter import messagebox
import sys
import os

def test_tkinter():
    print("Creating tkinter window...")
    
    # Try to bring Python to foreground (macOS specific)
    try:
        os.system('''osascript -e 'tell application "Python" to activate' ''')
    except:
        pass
    
    try:
        os.system('''osascript -e 'tell application "Terminal" to activate' ''')
    except:
        pass
    
    root = tk.Tk()
    root.title("Tkinter Test - Can you see this?")
    root.geometry("500x300+100+100")  # width x height + x_offset + y_offset
    
    # Force window to front
    root.lift()
    root.attributes('-topmost', True)
    
    # Create content
    frame = tk.Frame(root, bg='lightblue', relief='raised', bd=5)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    label = tk.Label(frame, text="TEST WINDOW", font=('Arial', 24, 'bold'), 
                    bg='lightblue', fg='red')
    label.pack(pady=20)
    
    info_label = tk.Label(frame, text="If you can see this, tkinter is working!", 
                         font=('Arial', 12), bg='lightblue')
    info_label.pack(pady=10)
    
    def close_window():
        print("Button clicked! Window will close.")
        messagebox.showinfo("Success", "Button works! tkinter is functioning.")
        root.quit()
    
    button = tk.Button(frame, text="Click Me to Close", command=close_window,
                      font=('Arial', 14), bg='yellow', padx=20, pady=10)
    button.pack(pady=20)
    
    # Remove topmost after a delay
    root.after(1000, lambda: root.attributes('-topmost', False))
    
    print("Window created. Starting mainloop...")
    print(f"Window geometry: {root.geometry()}")
    print("If you don't see a window, there may be a display/focus issue.")
    
    root.mainloop()
    print("Window closed.")

if __name__ == "__main__":
    test_tkinter()