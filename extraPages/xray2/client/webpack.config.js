
const path = require('path');

module.exports = {
	mode: 'development',
	entry: {
		index: './src/index.js',
	},
	devtool: 'inline-source-map',
	// devServer: {
	// static: './public',
	// },
	output: {
		filename: '[name].bundle.js',
		path: path.resolve(__dirname, 'public'),
		// clean: true,
		publicPath: '/',
	},
};
