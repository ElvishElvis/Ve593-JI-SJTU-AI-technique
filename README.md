# `VE593`

***This is a repository for `Ve593` Problem solving with AI technique by Prof. Paul Weng.***
___
## Catalog

* [Warning](#warning)
* [Project 1](#project1)
* [Project 2](#project2)
* [Project 3](#project3)
* [Project 4](#project4)

___

## <a name = "project1" />Project 1:

The project description is in [Project-1.pdf].

For [part1], you are required to construct seven Graph Searching Algorithms.

Unweighted Graph:

* [Breadth-First Search (BFS)]
* [Depth-First Search (DFS)]
* [Depth-Limited Search (DLS)]
* [Iterative Deepening Search (IDS)] 

PS: DLS and IDS are quite similar, thus they refer to the same Wiki page

Weighted Graph:

* [Uniform-Cost Search (UCS)] (or the Dijkstra Algorithm)
* [A Star Search (A*)]
* [Monte Carlo Tree Search (MCTS)]

A useful slides can be found [here].

You can find the codes of algorithm in [search.py].



For [part2], you are required to implement the game [Clickomania] and apply Graph Searching Algorithms you had done in part1 to find the optimal solution(or trying to find).

You can find two ways to implement the game in [Game1] and [Game2] and two ways to implement the player in [Player1] and [Player2]

[here]: https://wenku.baidu.com/view/396d792731b765ce050814df.html
[part1]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/tree/master/Project%201/Part1
[part2]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/tree/master/Project%201/Part2
[Game2]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%201/Part2/clickomania11111.py
[Game1]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%201/Part2/clickomania.py
[Player1]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%201/Part2/clickomaniaplayer.py
[Player2]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%201/Part2/BFS%20Player.py


[Clickomania]: http://www.8games8.com/puzzle/clickomania
[Project-1.pdf]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%201/Project-1.pdf
"project1"
[search.py]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%201/Part1/search.py
[Breadth-First Search (BFS)]: https://en.wikipedia.org/wiki/Breadth-first_search
[Depth-First Search (DFS)]: https://en.wikipedia.org/wiki/Depth-first_search
[Iterative Deepening Search (IDS)]: https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
[Depth-Limited Search (DLS)]: https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
[Uniform-Cost Search (UCS)]: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Practical_optimizations_and_infinite_graphs
[A Star Search (A*)]: https://en.wikipedia.org/wiki/A*_search_algorithm
[Monte Carlo Tree Search (MCTS)]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
[figure_1.png]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%201/figure_1.png


## <a name = "project2" />Project 2:
The project description is in [p2-description].

[p2-description]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%202/Project-2.pdf
"project2"

In part 1, you are required to construct a [Bayesian network (BN)] and implement Structure Learning Parameter Learning and Inference

[Bayesian network (BN)]: https://en.wikipedia.org/wiki/Bayesian_network


In part2, you are required to use the BN to testify on two dataset, [wine] and [protein].

[wine]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%202/wine.csv
[protein]: https://github.com/ElvishElvis/Ve593-JI-SJTU-AI-technique/blob/master/Project%202/protein.csv



## <a name = "project3" />Project 3:

The project description is in [p3-description].

[p3-description]: https://github.com/Tom-Pomelo/VE281/blob/master/Project3/Programming-Assignment-Three-Description.pdf

You are required to construct three priority queues and then solve a minimal-path-weight problem.

* [binary heap]
* [unsorted heap]
* [fibonacci heap]

[binary heap]: https://en.wikipedia.org/wiki/Binary_heap
[unsorted heap]: https://en.wikipedia.org/wiki/Heap_(data_structure)
[fibonacci heap]: https://en.wikipedia.org/wiki/Fibonacci_heap

You can find the codes in [p3-implementation].

[p3-implementation]: https://github.com/Tom-Pomelo/VE281/tree/master/Project3/project3/project3

## <a name = "project4" />Project 4:
The project description is in [p4-description].

You are required to construct some efficient and effective data structures to model a equity-transaction system.

Some feasible implementation may contain:

* [vector]
* [map]
* [multi-map]
* [set]
* [multi-set]

[p4-description]: https://github.com/Tom-Pomelo/VE281/blob/master/Project4/Programming-Assignment-Four.pdf

[vector]: https://en.wikipedia.org/wiki/Sequence_container_(C%2B%2B)#Vector
[map]: https://en.wikipedia.org/wiki/Associative_array
[multi-map]: https://en.wikipedia.org/wiki/Multimap
[set]: https://en.wikipedia.org/wiki/Set_(abstract_data_type)
[multi-set]: https://en.wikipedia.org/wiki/Multiset

You can find the codes in [p4-implementation].

[p4-implementation]: https://github.com/Tom-Pomelo/VE281/blob/master/Project4/main.cpp

## <a name = "project5" />Project 5:

The project description is in [p5-description].

[p5-description]: https://github.com/Tom-Pomelo/VE281/blob/master/Project5/Programming-Assignment-Five-Description.pdf

You are required to solve the Directed-Acyclic-Graph-Decision, Shortest-Path and Minimal-Spanning-Tree problems. 

Some feasible algorithms could be 

* [topology sorting]
* [Dijkstra's algorithm]
* [Bellman-Ford Algorithm] 
* [Prim’s Algorithm] 
* [Kruskal’s Algorithm]

[topology sorting]: https://en.wikipedia.org/wiki/Topology
[Dijkstra's algorithm]: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
[Bellman-Ford Algorithm]: https://en.wikipedia.org/wiki/Bellman–Ford_algorithm
[Prim’s Algorithm]: https://en.wikipedia.org/wiki/Prim%27s_algorithm
[Kruskal’s Algorithm]: https://en.wikipedia.org/wiki/Kruskal%27s_algorithm

You can find the codes in [p5-implementation].

[p5-implementation]: https://github.com/Tom-Pomelo/VE281/blob/master/Project5/main.cpp

## <a name = "warning" />Warning:
For students who are currently taking `Ve593`, please abide by the ***Honor Code*** in case of unnecessary troubles. 

### CopyRight All Rights Reserved
