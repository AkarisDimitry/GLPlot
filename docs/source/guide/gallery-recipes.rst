Gallery recipes and style presets
==================================

The workstation turns the example gallery into something you can use on your own data, and shows
every look it can apply as a real render of itself. Both live in panels:

* **Gallery** (``Ctrl+Shift+L``) -- 22 *recipes*: pick one, say which column goes where, and the plot is
  drawn on the scene in the look it was designed in.
* **Style** > *Styles* -- 25 presets, each shown on a sample plot of your choice.

Recipes
-------

A recipe is a kind of plot, the columns it reads, and a look. Open the Gallery panel, choose a
dataset (paste a table into the Data panel first, or press **Try with sample data**), pick a card:

1. The columns are guessed from their names (``time``, ``signal``, ``x``, ``height`` ...); change any
   of them in the *Columns* section. A single column is plotted against its row number, and
   ``(index)`` can stand in for any column.
2. The large picture is a **preview of your data**, rendered in the background by the headless
   export. It tells you whether the mapping makes sense; it is not what the scene will show.
3. *Look* chooses the page and colours: the recipe's own look, *Keep my current look*, or any
   preset.
4. **Apply to scene** draws it as one undo step. If the data cannot be drawn the button is
   disabled and the reason is shown (for example, a surface needs ``x`` and ``y`` on a full grid).

====================  ===========================================================================
Category              Recipes
====================  ===========================================================================
Lines                 signal trace, step response, event stems, filled area, line family
Scatter               colour-mapped scatter, bubble chart
Density               2-D density heat map, hexagon density, histogram
Bars                  bar chart
Fields                matrix heat map, field from scattered samples
3D                    point cloud, volumetric cloud, surface, wireframe, 3-D bars, vector
                      field, trajectory, 3-D stems, ribbon
====================  ===========================================================================

Each card names the ``examples/gallery`` script it was distilled from.

What a recipe is not
^^^^^^^^^^^^^^^^^^^^

* **Not a live link.** The layers are copies of the data; edit a cell and apply again to update
  them. (Binding a derived layer to its table is the Data panel's job.)
* **Not a restyle of your existing layers.** Applying a recipe sets the page (background, grid,
  effects) and colours the layers *it creates*; layers you already had are left alone, and undo
  removes the recipe's layers and puts the page, the axis names and the 2-D/3-D mode back.

From code
^^^^^^^^^

Recipes are plain data in :mod:`glplot.gui.recipes`::

   from glplot.gui import recipes

   recipe = recipes.get_recipe("starfield")
   recipes.assign_columns(recipe, ["x_pc", "y_pc", "age"])   # {'x': 'x_pc', 'y': 'y_pc', 'c': 'age', ...}
   table = recipes.sample_table(recipe)                      # the card's own example data
   rgb = recipes.render_preview_rgb(recipe)                  # headless render, (H, W, 3) uint8

Style presets
-------------

Each preset sets the page, the palette, the grid, line weight, effects, the 3-D axis box and --
new -- **a colormap** for anything coloured by value (an image, a colour-by-column scatter, a
surface). The six new looks are:

=============  ===================================================================================
aurora         polar-night gradient, mint and ice lines with a soft bloom
synthwave      ultraviolet night, magenta grid, cyan and orange glow
sunset         plum-to-ember gradient, orange, gold and coral
ocean          deep-water gradient, aqua and seafoam with a coral accent
vintage        aged paper with fine grain, sepia ink, mid-century pigments
candy          thick strokes, sticker-outlined markers, sweet-shop colours on pink-white
=============  ===================================================================================

The Style panel's cards are real renders. **Preview on** switches the sample plot (lines, scatter,
bars, heat map, 3-D surface), because one look reads very differently on each. Two switches control
what a click does to existing layers:

* *Restyle existing layers* -- repaint uniform colours with the preset's palette, line width and
  point size.
* *Use the style's colormap* -- re-map layers coloured by value through the preset's colormap. Their
  data is untouched; only the ramp changes, and undo restores it exactly.

From code, ``gplt.plot_style("synthwave")`` does both, and leaves the preset's colormap as the
figure's default for what it draws next (an explicit ``cmap=`` or ``gplt.set_cmap()`` still wins;
``colormap=False`` leaves colormaps alone).

Only the *luminous* looks (``neon``, ``aurora``, ``synthwave``) bloom. Glow is additive light and
costs legibility, so it stays out of the paper, presentation and workstation looks.

Regenerating the thumbnails
---------------------------

The thumbnails are committed (``glplot/gui/assets/gallery``, about 1.4 MB). They are captured from
the real OpenGL window, so regenerating them needs a display and a GPU::

   python tools/build_gallery_assets.py

Run it after adding a preset or a recipe; a test fails until the committed set matches the
catalogue. See ``tools/README.md`` for the options (single styles, a contact sheet, re-encoding
without GL).
