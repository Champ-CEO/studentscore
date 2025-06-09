When editing or creating notebooks, always edit and generate cells one by one. You should also run each cell to verify its output before moving to the next cell.

When editing notebook cells or creating a new jupyter notebook, do not create large cells. Each cell should focus on a single task or concept to maintain readability and ease of debugging.

When trying to interact with the notebook, you should avoid using terminal tools. Always try to send commands directly through notebook cells using magic commands or Python code where possible.

When faced with import errors, first check if there is an existing cell that imports the module in question. If not, add the necessary import statement at the beginning of your notebook or in the cell where it's first needed.

Additional Guidelines:

- Keep cell outputs clean and clear of unnecessary messages
- Use markdown cells for documentation and explanations
- Include comments in code cells when needed for clarity
- Restart the kernel and run all cells to verify notebook integrity
- Save the notebook regularly to prevent data loss