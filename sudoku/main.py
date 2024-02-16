class cell():
    def __init__(self, number) -> None:
        self.number = number
        self.value = None
        self.possible = [1,2,3,4,5,6,7,8,9]
        pass

    def fill(self):
        if len(self.possible) == 1:
            self.sett(self.possible[0])

    def remove(self, waarde):
        if waarde in self.possible:
            self.possible.remove(waarde)
            self.fill()

    def sett(self, waarde):
        self.value = waarde
        if waarde != 0:
            self.possible = []
            self.update_group()

    def update_group(self):
        for r in rows[self.number//9]:
            r.remove(self.value)
        for c in columns[self.number%9]:
            c.remove(self.value)
        for s in squares[self.number//27][int(self.number%9/3)]:
            s.remove(self.value)

class row():
    def __init__(self, cells: list[cell]) -> None:
        self.possible = [1,2,3,4,5,6,7,8,9]
        self.containing = cells
        pass

    def update_cells(self, waarde):
        if waarde in self.possible:
            self.possible.remove(waarde)
        for i in self.containing:
            i.remove(waarde)
        

def grid():
    getallen = []
    for i in cells:
        if i.value is None:
            getallen.append(0)
        getallen.append(i.value)
    for i in range(9):
        print(f"{getallen[i*9:i*9+3]}|{getallen[i*9+3:i*9+6]}|{getallen[i*9+6:i*9+9]}")
    
cells = [cell(i) for i in range(9*9)]

rows = [[cells[i+j*9] for i in range(9)] for j in range(9)]

columns = [[cells[i*9+j] for i in range(9)] for j in range(9)]

squares = [[[cells[i%3+(i//3*9)+j*3+k*27] for i in range(9)] for j in range(3)] for k in range(3)]

eigen_regel = "007106040502000018004090703200389000000004000040061050170000080008010500035000260"
for i in range(81):
    cells[i].sett(int(eigen_regel[i]))

grid()

board1 = """
[3, 8, 7]|[1, 2, 6]|[9, 4, 5]
[5, 9, 2]|[0, 0, 0]|[6, 1, 8]
[6, 1, 4]|[0, 9, 0]|[7, 2, 3]
[2, 0, 0]|[3, 8, 9]|[0, 7, 0]
[0, 0, 0]|[0, 0, 4]|[0, 0, 0]
[0, 4, 0]|[0, 6, 1]|[0, 5, 0]
[1, 7, 0]|[0, 0, 0]|[0, 8, 0]
[0, 0, 8]|[0, 1, 0]|[5, 0, 0]
[0, 3, 5]|[0, 0, 0]|[2, 6, 0]"""

board2 = """
[3, 8, 7]|[1, 2, 6]|[9, 4, 5]
[5, 9, 2]|[0, 0, 0]|[6, 1, 8]
[6, 1, 4]|[0, 9, 0]|[7, 2, 3]
[2, 0, 0]|[3, 8, 9]|[0, 7, 0]
[0, 0, 0]|[0, 0, 4]|[0, 0, 0]
[0, 4, 0]|[0, 6, 1]|[0, 5, 0]
[1, 7, 0]|[0, 0, 0]|[0, 8, 0]
[0, 0, 8]|[0, 1, 0]|[5, 0, 0]
[0, 3, 5]|[0, 0, 0]|[2, 6, 0]"""

print(board1==board2)