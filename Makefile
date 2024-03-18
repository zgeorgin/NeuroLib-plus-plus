test: test.cpp perceptrone.o
	g++ -o test test.cpp perceptrone.o
perceptrone.o: perceptrone.cpp
	g++ -c -o perceptrone.o perceptrone.cpp