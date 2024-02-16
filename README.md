# Projects
 In this repositiry you can find code that will find the solution to your games problem.

## Games
 - [Minesweeper](#Minesweeper)
 - [Sudoku](#Sudoku)

## Minesweeper
 **This is not the final product, it can not yet comlepete the hardest level of google minesweeper.**

 In this code a screenshot is taken at place x,y of width, height. You will need to change these variables. The AI looks at each cell and looks for the color of the numbers.

 The AI will go and look if there are the same amount of green sqaures and flags around the cell as the number in it, at that the AI adds the location of the green sqaures around to a list that needs to be right clicked.
 If there are the same amount of flags around a cell as the number, the location of all green sqaures around will be added to a list witch will be left clicked.

## Sudoku
 In the code you will find a variable called "regel" this variable needs a string of 81 numbers, make the string starting at the top left corner of the sudoku going left until the end of the row, rown a row. If a cell does not have a value write it as down a "0".

 The code will return a filled in version of the sudoku, it will be printed in the terminal.