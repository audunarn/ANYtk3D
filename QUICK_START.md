# ANYtk3D quick start

A short walkthrough of the example in the [README quick start](README.md#quick-start).

1. Create the Tk root window and a `Tkinter3DCanvas` inside it, e.g.
   `Tkinter3DCanvas(root, width=900, height=600, bg='white')`.
2. Pack the canvas with `canvas.pack(fill='both', expand=True)` so it fills the window.
3. Add shapes such as `add_box`, `add_sphere`, `add_tube`, `add_beam` and
   `add_grid`, using `Point3D` for centers and endpoints.
4. Call `canvas.fit_to_scene()` once to frame everything you added.
5. Start the Tk event loop with `root.mainloop()`.

See the full runnable snippet and the retained-arrays section in the README.
