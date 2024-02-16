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

    def update_group(self):
        for r in rows[self.number//9]:
            r.remove(self.value)
        for c in columns[self.number%9]:
            c.remove(self.value)
        for s in squares[self.number//27][int(self.number%9/3)]:
            s.remove(self.value)


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

eigen_regel = "000020345008000007050094008039400760000309400060075083042951800085730010071248039"
for i in range(81):
    cells[i].sett(int(eigen_regel[i]))

for i in range(100):
    done = True
    for j in cells:
        if j.value != 0:
            j.update_group()
        else:
            done = False
    if done:
        print("klaar")
        break

grid()
