.PHONY: configure build run-vector-add env clean

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build:
	cmake --build build -j

run-vector-add:
	./build/bin/lesson_01_vector_add 1000000

env:
	python3 scripts/record_env.py --output profiler_reports/environment.json

clean:
	rm -rf build
