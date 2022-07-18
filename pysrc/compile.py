import os, sys, re
import dominate, dominate.util, dominate.tags as D


def formatCode(code, lang):
    from pygments.lexers import guess_lexer, guess_lexer_for_filename, get_lexer_by_name
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    if lang == '':
        lexer = guess_lexer(code)
    else:
        lexer = get_lexer_by_name(lang)
    return highlight(code, lexer, HtmlFormatter())

def getHeaderClass(l, *a,**k):
    l = max(min(int(l),6),1)
    return getattr(D,'h'+str(l))

def processLine(line):
    # Handle image

    # Handle link
    match = re.search('\[(.*?)\]\(([^ ()\[\]]*?)\)', line)
    if match:
        txt, link = match.groups()
        # line = line[:match.start()] + '<a href="{}">{}</a>'.format(link, txt) + line[match.end():]
        # line = D.span(line[:match.start()]) + D.a(txt, href=link) + D.span(line[match.end():])
        # return processLine(line)
        out = []
        if match.start() > 0: out.append(processLine(line[:match.start()]))
        out.append(D.a(txt,href=link))
        if match.end() < len(line): out.append(processLine(line[match.end():]))
        return out

    # Backtick highlight span
    match = re.search('`([^`]+)`', line)
    if match:
        out = []
        if match.start() > 0: out.append(processLine(line[:match.start()]))
        out.append(D.span(match.groups()[0], _class='backticked'))
        if match.end() < len(line): out.append(processLine(line[match.end():]))
        return out

    else:
        # return line
        return [D.span(line)]

def parseMarkdownToHtml(title, lines):
    i = 0
    doc = dominate.document(title=title)
    with doc.head:
        D.link(rel='stylesheet', href='res/main.css')
        D.link(rel='stylesheet', href='res/boostrap.darkly.css')
        D.link(rel='stylesheet', href='res/pygments.css')
        #script(type='text/javascript', src='script.js')

    with doc.body as body:
        body.add(dominate.util.raw(get_navbar()))
        body.add(dominate.util.raw(get_mainImage()))

        with D.div(_class='container') as container:

            with D.div(_class='page-header'):
                with D.div(_class='col-lg-8 col-md-7 col-sm-6'):
                    pass

            while i < len(lines):
                line = lines[i]
                sline = line.strip()

                matchOl = re.search('^[\W]*\d+[.)]', line)

                if len(sline) == 0:
                    i += 1

                # Header
                elif sline[0] == '#':
                    lvl = 0
                    while sline[lvl] == '#': lvl+=1
                    txt = sline[lvl:].strip()
                    getHeaderClass(lvl)(processLine(txt))
                    i += 1


                # begin ul
                elif sline[0] == '-':
                    with D.ul() as list:
                        while i < len(lines):
                            line = lines[i]
                            match = re.search('^[\W]*[-+]', line)
                            if not match: break
                            txt = line[match.end():]
                            li = D.li(processLine(txt))
                            list += li
                            i += 1

                # begin ol
                elif matchOl:
                    with D.ol() as list:
                        while i < len(lines):
                            line = lines[i]
                            matchOl = re.search('^[\W]*\d+[.)]', line)
                            if not matchOl: break
                            txt = line[matchOl.end():]
                            li = D.li(processLine(txt))
                            list += li
                            i += 1

                # Begin code block
                elif sline[0:3] == '```':
                    language = sline[3:]
                    code = ''
                    while True:
                        i += 1
                        if i >= len(lines): break
                        line = lines[i]
                        sline = line.strip()
                        if sline[:3] == '```':
                            i += 1
                            break
                        else: code += '\n' + line
                    fcode = formatCode(code,language)
                    fcode = fcode.replace('\n\n','\n')
                    D.div(dominate.util.raw(fcode), _class='code')

                # Begin paragraph
                else:
                    # lst = processLine(line)
                    # Remove outer span, if output is just a span
                    # if len(lst) == 1 and isinstance(lst[0], D.span): D.p(lst[0][0])
                    D.p(processLine(line))
                    i += 1

    return doc.render()


    '''
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.lexers.markup import MarkdownLexer
    from pygments.formatters import HtmlFormatter
    code = '\n'.join(lines)
    return highlight(code, MarkdownLexer(), HtmlFormatter())
    '''



def makePygmentsCss():
    from pygments.formatters import HtmlFormatter
    with open('res/pygments.css', 'w') as fp:
        print(HtmlFormatter(style='inkpot').get_style_defs('.highlight'), file=fp)

def get_mainImage():
    # return '\n<image src="res/mainImage.jpg" id="mainImage"/>\n'
    return '\n<div  id="mainImage"></div>\n'

def get_navbar():
    return '''
<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
  <div class="container-fluid">
    <a class="navbar-brand" href="#">Navbar</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarColor02" aria-controls="navbarColor02" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>

    <div class="collapse navbar-collapse" id="navbarColor02">
      <ul class="navbar-nav me-auto">
        <li class="nav-item">
          <a class="nav-link active" href="#">Home
            <span class="visually-hidden">(current)</span>
          </a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">Features</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">Pricing</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">About</a>
        </li>
        <li class="nav-item dropdown">
          <a class="nav-link dropdown-toggle" data-bs-toggle="dropdown" href="#" role="button" aria-haspopup="true" aria-expanded="false">Dropdown</a>
          <div class="dropdown-menu">
            <a class="dropdown-item" href="#">Action</a>
            <a class="dropdown-item" href="#">Another action</a>
            <a class="dropdown-item" href="#">Something else here</a>
            <div class="dropdown-divider"></div>
            <a class="dropdown-item" href="#">Separated link</a>
          </div>
        </li>
      </ul>
    </div>
  </div>
</nav>
    '''

def main():
    makePygmentsCss()

    dir = 'posts'
    for file in os.listdir(dir):
        path = os.path.join(dir,file)
        if os.path.isdir(path):
            pass
        else:
            day,month,year,title = file.split('_', 3)

            title = title.replace('.md','')
            title = ' '.join(s.capitalize() for s in title.split('_'))

            with open(path,'r') as fp: lines = fp.readlines()
            src = parseMarkdownToHtml(title, lines)
            with open('out.html', 'w') as fp: print(src,file=fp)


if __name__ == '__main__':
    main()