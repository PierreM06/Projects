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
 When the code is run a window will be made with a 9x9 grid filled with cells, each cell has 9 buttons one for each possible number. When one of the buttons is clicked the buttons will disappear and the number will fill the cell. The button of that number will be removed from the other cells in that row, column and square. 
 
 When there is only one number left in the cell the cell will be assigned that number so you should never see a single number in a cell. 
 A cell will alse be assigned a number when it is the only cell in a row, column or square that can have that can still be assigned that number.

 The code will return a filled in version of the sudoku, it will be printed in the terminal.