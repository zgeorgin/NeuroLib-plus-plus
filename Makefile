test: test.cpp perceptrone.o
	g++ -o test test.cpp perceptrone.o
perceptrone.o: source/perceptrone.cpp
	g++ -c -o perceptrone.o source/perceptrone.cpp