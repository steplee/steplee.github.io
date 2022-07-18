const fs = require('fs');
const crypto = require('crypto');

function generateKey(outName) {
	crypto.webcrypto.subtle.generateKey({'name': 'AES-CBC', length: 192}, true, ["encrypt", "decrypt", "wrapKey", "unwrapKey"]).then(key => {
		crypto.webcrypto.subtle.exportKey('jwk', key).then(exKey => {
			console.log(' - Generated key')

			fs.writeFile(outName, JSON.stringify(exKey), err => {
				if (err) {
					console.error('error writing file', outName);
					console.error(err);
				} else
					console.log(' - wrote to', outName)
			});
		});
	})
}

function generateKeyFromCli() {
	const argv = process.argv;
	var outName = '';
	for (i=0; i<argv.length; i++) {
		if (argv[i] == '--out') {
			i++;
			outName = argv[i];
		}
	}

	if (outName == '') {
		console.error(' - you must pass like "--out asd.key"')
		process.exit(1)
	}

	generateKey(outName)
}


for (arg of process.argv) {
	if (arg == "gen" || arg == "genKey")
		generateKeyFromCli()
}
