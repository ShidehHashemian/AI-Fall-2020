/*----------     First Question: List's Reverse     ----------*/
reverse_list(List,Reverse):-
    reverse(List,Reverse).

/*----------     Second Question: Quick Sort     ----------*/
partition_list(_, [], [], []).% if there is nothing to partition. (end of recusion)

partition_list(Pivot, [Head|Tail], [Head|LessOrEqualThan], GreaterThan) :-
    Pivot >= Head, % then head is in the list in which values are less or equal to the pivot
    partition_list(Pivot, Tail, LessOrEqualThan, GreaterThan). %partition the rest of the list
partition_list(Pivot, [Head|Tail], LessOrEqualThan, [Head|GreaterThan]) :-
    % it execute this rule if pivot< Head, then head should belong to GreaterThan
    partition_list(Pivot, Tail, LessOrEqualThan, GreaterThan).% partition the rest of the list

quick_sort([], []). %if the given list empty, there is nothing to sort(end of recursion)
quick_sort([Head|Tail], Sorted) :-
    partition_list(Head, Tail, Low, High),% use fist value in the list as a pivot and partition the given list based on that
    % now sort the given list after partitioning
    quick_sort(Low, SortedLow),
    quick_sort(High, SortedHigh),
    %conquer the sorted lists with the pivot
    append(SortedLow, [Head|SortedHigh], Sorted),!.

/*----------     Third Question: Family Relations     ----------*/

% fist define some facts to use them for examples
male(oliver).
male(jack).
male(harry).
male(jacob).
male(noah).
male(liam).

female(olivia).
female(emma).
female(sophia).
female(isabella).
female(amelia).
female(charlotte).

parent_of(harry,jacob).
parent_of(sophia,jacob).

parent_of(oliver,olivia).
parent_of(oliver,jack).
parent_of(emma,olivia).
parent_of(emma,jack).

parent_of(jacob,noah).
parent_of(jacob,amelia).
parent_of(olivia,noah).
parent_of(olivia,amelia).

parent_of(jack,isabella).
parent_of(jack,liam).
parent_of(charlotte,isabella).
parent_of(charlotte,liam).

% now use the facts to execute rules
wife_of(X,Y):-
    mother_of(X,Z),
    father_of(Y,Z).

husband_of(X,Y):-
    father_of(X,Z),
    mother_of(Y,Z).

father_of(X,Y):-
    male(X),
    parent_of(X,Y).

mother_of(X,Y):-
    female(X),
    parent_of(X,Y).

fatherinlaw_of(X,Y):-
    father_of(X,Z),
    parent_of(Z,W),
    parent_of(Y,W),!.

motherinlaw_of(X,Y):-
    mother_of(X,Z),
    parent_of(Z,W),
    parent_of(Y,W),!.


daughter_of(X,Y):-
    female(X),
    parent_of(Y,X).

son_of(X,Y):-
    male(X),
    parent_of(Y,X).


sister_of(X,Y):-
    female(X),
    mother_of(Z,X),
    mother_of(Z,Y),
    X\=Y.

brother_of(X,Y):-
    male(X),
    mother_of(Z,X),
    mother_of(Z,Y),
    X\=Y.

grandparent_of(X,Y):-
    parent_of(X,Z),
    parent_of(Z,Y).

grandfather_of(X,Y):-
    male(X),
    grandparent_of(X,Y).


grandmother_of(X,Y):-
    female(X),
    grandparent_of(X,Y).

aunt_of(X,Y):-
    parent_of(Z,Y),
    sister_of(X,Z).

uncle_of(X,Y):-
    parent_of(Z,Y),
    brother_of(X,Z).

cousin_of(X,Y):-
    aunt_of(Z,X),
    parent_of(Z,Y).

cousin_of(X,Y):-
    uncle_of(Z,X),
    parent_of(Z,Y).

nephew_of(X,Y):-
    male(X),
    sister_of(Z,Y),
    parent_of(Z,X).

nephew_of(X,Y):-
    male(X),
    brother_of(Z,Y),
    parent_of(Z,X).

niece_of(X,Y):-
    female(X),
    sister_of(Z,Y),
    parent_of(Z,X).

niece_of(X,Y):-
    female(X),
    brother_of(Z,Y),
    parent_of(Z,X).

grandson_of(X,Y):-
    son_of(X,Z),
    parent_of(Y,Z).


granddaughter_of(X,Y):-
    daughter_of(X,Z),
    parent_of(Y,Z).
