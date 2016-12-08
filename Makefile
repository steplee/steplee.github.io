
all:
	cd res; \
	stack exec site build; \
	cp _site/* ../ -rf; \
	cd ..; \
	rm -rf ./_cache

clean:
	rm -rf index.html posts about.html contact.markdown _site _cache archive.html images css js res/_site res/_cache

