
all:
	cd res; \
	stack runghc ../site.hs build; \
	cp _site/* ../ -r \

clean:
	rm -rf index.html posts about.html contact.markdown _site _cache archive.html images css js res/_site res/_cache

