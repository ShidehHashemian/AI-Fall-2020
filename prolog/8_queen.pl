:- use_module(library(clpfd)).

n_queens(N, Qs) :-
        length(Qs, N),
        Qs ins 1..N,% check if each cell value is valid (between 1-N or not)
        safe_queens(Qs).% check if each cell value is valid (between 1-N or not)

safe_queens([]).% in empty array which represent chess board, no queen attacks any
safe_queens([Q|Qs]) :-
    safe_queens(Qs, Q, 1),% check if the queen in the first column attacks others
    safe_queens(Qs).% repeat for others
safe_queens([], _, _).
safe_queens([Q|Qs], Q0, D0) :-%Q0: the queen that we check if attacks any other or not D0: the column of this queen
    Q0 #\= Q,% check horizontal attacks
    abs(Q0 - Q) #\= D0, % check diagonal attacks
    D1 #= D0 + 1,% update D for the next column queen
    safe_queens(Qs, Q0, D1).% call it for the next column queen

solution(Qs):-% look for a solution
    n_queens(8,Qs), % check if it satisfy our goal or not
    maplist(between(1,8),Qs).% generate different array that each represents a board

