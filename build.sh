#!/bin/bash

`npm bin`/browserify --debug extraPages/lieViz/lieViz.js > extraPages/lieViz/lieViz.bundled.js

# python3 -m pysrc.compile
