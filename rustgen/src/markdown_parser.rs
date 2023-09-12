use crate::syntax_highlight::SyntaxHighlighter;

use std::collections::BTreeMap;

/*

   Markdown is simple enough that we don't really need to seperate lexing from parsing.

   My original impl had nearly no nesting, but you kind of want paragraphs wrapped in html <p>'s, so
   I had to have a nested type for that (as well as for lists)

   There are parts of this that are very inefficent, mainly because each try_* call can only output
   one item. Therefore when we are in try_normal() and we see a possible quoted section, we have check
   if it is valid until the EOL and stop if it is.
   Then the next loop we have to detect the quoted section in try_ticked() all again.

   A better parser would handle without this repetition by recursive descent etc.

   FIXME: Do not support ul/ol yet, nor '>' quotes

*/

pub struct ParsedMarkdownDocument {
    pub meta: BTreeMap<String, String>,
    pub html: String,
    pub src_file: Option<String>,
}

struct MarkdownParser<'a> {
    sh: Option<&'a SyntaxHighlighter>
}

#[derive(Debug)]
enum Tok {
    Metadata(Vec<(String,String)>), // key-value pairs
    Section(u8, Vec<Tok>),
    Ticked(String),
    TripleTicked(String,String), // (language, code)
    Underscored(String),
    OneStarred(String),
    TwoStarred(String),
    Normal(String),
    Paragraph(Vec<Tok>),
    Link(String, String), // (label, src)
    Image(String, String), // (label, src)
    Break(),
    Ul(Vec<Tok>), // recursive
    Ol(Vec<Tok>), // recursive
}

//
// str slices work by using byte offsets, but asserting that 
// any slice lies on a utf8-char boundary.
//
struct Stream<'a> {
    s: &'a str,
    i: usize,
}
// Lifetime 'b should be contained within 'a
struct StreamUse<'a,'b> {
    strm: &'b mut Stream<'a>,
    j: usize,
}
impl<'a,'b> StreamUse<'a,'b> {
    fn from(strm: &'b mut Stream<'a>) -> StreamUse<'a,'b> {
        let j = strm.i;
        // let is_beginning_of_line_ = (j == 0) || (strm.s[j..j+1].chars().next().unwrap_or('.') == '\n');
        return StreamUse{strm, j};
    }

    fn commit(&mut self) -> () {
        self.strm.i = self.j;
    }

    fn next(&mut self) -> Option<char> {
        let c = self.strm.s[self.j..].chars().next()?;
        self.j += c.len_utf8();
        return Some(c);
    }

    fn peek(&mut self) -> Option<char> {
        let c = self.strm.s[self.j..].chars().next()?;
        return Some(c);
    }

    // WARNING: chars().count() may need to be chars().as_str().len() ...

    fn read_until_space(&mut self) -> Option<&str> {
        let oj = self.j;
        let n = self.strm.s[self.j..].chars().take_while(|&c| { c!=' ' && c!='\t' && c!='\n' }).count();
        // println!(" - read until space : {} -> {}", self.j, self.j+n);
        self.j += n;
        return Some(&self.strm.s[oj..self.j]);
    }
    fn read_space(&mut self) -> Option<&str> {
        let oj = self.j;
        let n = self.strm.s[self.j..].chars().take_while(|&c| { c==' ' || c=='\t' || c=='\n' }).count();
        // println!(" - read space : {} -> {}", self.j, self.j+n);
        self.j += n;
        return Some(&self.strm.s[oj..self.j]);
    }
    fn read_until_end_of_line(&mut self) -> Option<&str> {
        // println!(" - read until eol : {}", self.j);
        let oj = self.j;
        let n = self.strm.s[self.j..].chars().take_while(|&c| { c!='\n' }).count();
        // println!("eol {} -> {}", self.j, self.j+n);
        self.j += n;
        return Some(&self.strm.s[oj..self.j]);
    }
}
impl<'a,'b> Drop for StreamUse<'a,'b> {
    fn drop(&mut self) -> () { }
}

fn strip(s: &str) -> &str {
    let mut a = 0;
    let mut b = s.len();
    let mut cs = s.chars();
    loop {
        if let Some(c) = cs.next() {
            if c == ' ' || c == '\t' {
                a += 1;
            } else {break}
        } else {break}
    }

    let mut cs = s.chars().rev();
    loop {
        if let Some(c) = cs.next() {
            if c == ' ' || c == '\t' {
                b -= 1;
            } else {break}
        } else {break}
    }
    b = s.len();
    return &s[a..b];
}

fn try_section(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    // Make sure we are at a newline symbol, but also allow being the first thing in the document.
    if start_j == 0 {
        if let Some(c) = user.peek() {
            if c != '#' { return None }
        }
    } else if let Some(c) = user.next() {
        if c != '\n' { return None }
    }

    // user.read_until_space();
    // user.read_space();
    if let Some(txt) = user.read_until_space() {
        if !(*txt).chars().all(|c| c=='#') {
            return None;
        }
        let n = txt.len();

        if n > 0 {
            user.read_space();

            // FIXME: ALLOW TICKS
            // let title = user.read_until_end_of_line().unwrap().to_owned();
            // user.commit();
            // return Some(Tok::Section(n as u8, vec![Tok::Normal(title)]));

            let mut out = vec![];
            let mut cur_normal = String::new();
            let mut cur_ticked = String::new();
            let mut is_ticked = false;
            loop {
                match user.next() {
                    Some('`') => {
                        if cur_normal.len() > 0 { out.push(Tok::Normal(cur_normal)); }
                        if cur_ticked.len() > 0 { out.push(Tok::Ticked(cur_ticked)); }
                        is_ticked = true;
                        cur_normal = String::new();
                        cur_ticked = String::new();
                    },
                    Some('\n') => {
                        if cur_normal.len() > 0 { out.push(Tok::Normal(cur_normal)); }
                        if cur_ticked.len() > 0 { out.push(Tok::Ticked(cur_ticked)); }
                        user.j -= 1;
                        user.commit();
                        return Some(Tok::Section(n as u8, out));
                    },
                    Some(c) => {
                        if  is_ticked { cur_ticked += &c.to_string(); }
                        if !is_ticked { cur_normal += &c.to_string(); }
                    },
                    None => {
                        return Some(Tok::Section(n as u8, out));
                    }
                }
            }


        }
    }

    None
}

fn try_ticked(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    if let Some(c) = user.next() {
        if c == '`' {
            while let Some(c) = user.next() {
                // If there is a newline before a closing tick, fail.
                if c == '\n' {
                    return None
                }
                if c == '`' {
                    user.commit();
                    // println!("emit ticked {} -> {}", start_j, user.j);
                    // return Some(Tok::Ticked(user.strm.s[start_j..user.j].to_owned()));
                    return Some(Tok::Ticked(user.strm.s[start_j+1..user.j-1].to_owned()));
                }
            }
        }
    }
    None
}

fn try_code(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    // Make sure we are at a newline symbol.
    if let Some(c) = user.next() {
        if c != '\n' { return None }
    }


    let c1 = user.next()?;
    let c2 = user.next()?;
    let c3 = user.next()?;
    if c1 != '`' || c2 != '`' || c3 != '`' { return None; }
    let lang = user.read_until_end_of_line().unwrap().to_owned();
    let nl = user.next().unwrap();
    assert!(nl == '\n');
    let start_j_code = user.j;

    let mut prev1 = '\0';
    let mut prev2 = '\0';
    loop {
        if let Some(c) = user.next() {
            if c == '`' && prev1 == '`' && prev2 == '`' {
                user.commit();
                return Some(Tok::TripleTicked(lang, user.strm.s[start_j_code..user.j-3].to_owned()));
            }
            prev2 = prev1;
            prev1 = c;
        } else {
            return None;
        }
    }
}

fn try_meta(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    // Make sure we are at a newline symbol.
    if start_j != 0 {
        if let Some(c) = user.next() {
            if c != '\n' { return None }
        }
    }

    let c1 = user.next()?;
    let c2 = user.next()?;
    let c3 = user.next()?;
    if c1 != '`' || c2 != '`' || c3 != '`' { return None; }
    let lang = user.read_until_end_of_line().unwrap().to_owned();

    // The lang must be empty, or 'meta'
    if !(lang.len() == 0 || lang == "meta") {
        return None;
    }
    // println!("LANG {lang}");

    let nl = user.next().unwrap();
    assert!(nl == '\n');
    let start_j_code = user.j;

    let mut kvs = vec![];

    let mut prev1 = '\0';
    let mut prev2 = '\0';
    loop {
        let line = user.read_until_end_of_line();
        if let Some(line) = line {
            if line.starts_with("```") {
                // user.next();
                while let Some(c) = user.peek() {
                    if c == '\n' { user.next(); }
                    else { break; }
                }
                user.j-=1;
                user.commit();
                return Some(Tok::Metadata(kvs));
            }
            else if let Some((k,v)) = line.split_once(":") {
                let k = strip(k);
                let v = strip(v);
                kvs.push((k.to_owned(),v.to_owned()));
                // println!("meta {k} {v}");
                user.next();
            } else {break}
        }
    }
    None
}

fn try_newline(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);

    if let Some(c) = user.next() {
        if c == '\n' {
            user.commit();
            return Some(Tok::Break());
        }
    }
    None
}

fn try_double_newline(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);

    if let Some(c) = user.next() {
        if c != '\n' {
            return None
        }
    }
    if let Some(c) = user.peek() {
        if c == '\n' {
            user.commit();
            return Some(Tok::Break());
        }
    }

    None
}

fn try_link_or_image(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    let mut is_image = false;

    if let Some(c) = user.peek() {
        if c == '!' {
            is_image = true;
            user.next();
        }
    }

    if let Some(c) = user.next() {
        if c != '[' { return None }
    }
    let open_bracket_j = user.j;
    let mut close_bracket_j = 0;
    let mut open_paren_j = 0;
    let mut close_paren_j = 0;

    let mut last_c = '\0';
    let mut state = 0;
    loop {
        if let Some(c) = user.next() {
            // println!("c={}",c);
            if c == '\n' {
                return None
            }
            if state == 0 && (c == ']' && last_c != '\\') {
                state = 1;
                close_bracket_j = user.j-1;
            }
            else if state == 1 && (c == '(' && last_c != '\\') {
                state = 2;
                open_paren_j = user.j;
            } else if state == 1 {
                // println!("failed link because non '](' sequence c={}",c);
                // return None
            }
            if state == 2 && (c == ')' && last_c != '\\') {
                state = 2;
                close_paren_j = user.j-1;

                let name = user.strm.s[open_bracket_j..close_bracket_j].to_owned();
                let link = user.strm.s[open_paren_j..close_paren_j].to_owned();

                user.commit();
                if is_image {
                    return Some(Tok::Image(name, link));
                } else {
                    return Some(Tok::Link(name, link));
                }
            }
            last_c = c
        } else { break }
    }
    None
}

// NOTE: This is for math blocks (inline math require no parsing)
fn try_mathblock(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    // Make sure we are at a newline symbol.
    if let Some(c) = user.next() {
        if c != '\n' { return None }
    }

    // FIXME: Also handle \[
    let c1 = user.next()?;
    let c2 = user.next()?;
    if c1 != '$' || c2 != '$' { return None; }
    // let nl = user.next().unwrap();
    // assert!(nl == '\n');
    let start_j_code = user.j;

    let mut prev1 = '\0';
    loop {
        if let Some(c) = user.next() {
            if c == '$' && prev1 == '$' {
                user.commit();
                return Some(Tok::Paragraph(vec![Tok::Normal(user.strm.s[start_j_code-2..user.j].to_owned())]));
            }
            prev1 = c;
        } else {
            return None;
        }
    }
}

fn try_paragraph(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    if let Some(c) = user.peek() {
        if c != '\n' { return None }
    }

    user.next(); // new line

    // Find first non-space. If there is atleast one space and if it is a number of it is is a dash, then this is NOT a paragraph.
    // FIXME: enable this and have try_list consumer at top-level.
    /*
    let mut num_spaces = 0;
    loop {
        if let Some(c) = user.peek() {
            if c == '\t' || c == ' ' {
                num_spaces += 1;
                user.next();
            } else {
                if num_spaces > 0 && (c == '-' || c.is_numeric()) {
                    return None;
                } else {
                    // This is a paragraph.
                    break;
                }
            }
        } else { break }
    }
    */

    // Parse inner elements
    let mut out = vec![];

    let mut strm2 = Stream{s: user.strm.s, i: user.j};

    for i in 0..999999 {

        if strm2.i >= strm2.s.len() { break; }
        if let Some(c) = strm2.s[strm2.i..].chars().next() {
            if c == '\n' { break; }
        }

        if let Some(tok) = try_ticked(&mut strm2) {
            out.push(tok);
            continue;
        }

        if let Some(tok) = try_link_or_image(&mut strm2) {
            out.push(tok);
            continue;
        }

        if let Some(tok) = try_normal(&mut strm2) {
            out.push(tok);
            continue;
        }

        // if let Some(tok) = try_newline(&mut strm2) {
            // out.push(tok);
            // break; // NOTE: Break
        // }

        panic!("invalid tok in paragraph {}/{}", strm2.i, strm2.s.len());
    }

    if strm2.i - user.strm.i == 1 {
        if let Some(c) = strm2.s[user.strm.i..].chars().next() {
            if c == '\n' { return None }
        }
    }

    // println!("push paragraph {} -> {} ({})", user.strm.i, strm2.i, &strm2.s[user.strm.i..strm2.i]);
    user.strm.i = strm2.i;
    return Some(Tok::Paragraph(out));
}

fn try_normal(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    // Can't start on \n
    if let Some(c) = user.peek() {
        if c == '\n' { return None }
    }

    let mut last_c = '\0';

    loop {
        if let Some(c) = user.next() {
            if c == '\n' {
                user.j -= 1;
                user.commit();
                return Some(Tok::Normal(user.strm.s[start_j..user.j].to_owned()));
            }

            if c == '`' {
                // Search until eol
                let j_at_open = user.j;
                if let Some(rest) = user.read_until_end_of_line() {
                    if rest.find('`').is_some() {
                        // This is valid, we have to backup and stop there.
                        // println!("normal back from tick {} {} {}", start_j, user.j, j_at_open-1);
                        user.j = j_at_open-1;
                        user.commit();
                        return Some(Tok::Normal(user.strm.s[start_j..user.j].to_owned()));
                    }
                }
                // No closing '`', so we can continue.
            }

            if c == '[' {
                // If last was '!', this is an inline image and the next lex call must see it.
                let j_at_open = if last_c == '!' { user.j-2 } else { user.j-1 };

                if let Some(rest) = user.read_until_end_of_line() {
                    if let Some(close_brace) = rest.find(']') {
                        let open_paren = rest[close_brace..].find("(");
                        let clos_paren = rest[close_brace..].find(")");

                        // This is valid, we have to backup and stop there.
                        if open_paren.is_some() && clos_paren.is_some() {
                            // '(' comes before ')', and we have a sequence that is exactly ']('
                            if (open_paren.unwrap() < clos_paren.unwrap()) && (open_paren.unwrap() == 1) {
                                user.j = j_at_open;
                                user.commit();
                                return Some(Tok::Normal(user.strm.s[start_j..user.j].to_owned()));
                            }
                        }

                    }
                }

                user.j = j_at_open+1;
            }
            /*
            if c == '[' {
                // let mut user1 = StreamUse::from(user.strm);
                let back = user.j-1;
                user.strm.i += back;
                println!("try because link");
                if let Some(_) = try_link_or_image(user.strm) {
                    println!("stop because link");
                    // user.j = back;
                    return Some(Tok::Normal(user.strm.s[start_j..user.j].to_owned()));
                }
                user.j = back;
            }
            */

            last_c = c;
        } else { break }
    }
    None
}

fn try_list(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    // Make sure we are at a newline symbol.
    if let Some(c) = user.next() {
        if c != '\n' { return None }
    }

    if let Some(c) = user.next() {
        // FIXME: Stopped here.
    }

    None
}


impl<'a> MarkdownParser<'a> {
    fn lex(&self, txt: &str) -> Vec<Tok> {
        use Tok::*;
        // let out = vec![Ticked(String::from("hello"))];
        let mut out = vec![];

        // Line-based
        /*
        let lines = txt.split("\n");
        for line in lines {
            if let Some(c) = line.chars().next() {
                if c == '#' {
                    let n = line.chars().take_while(|c| { *c == '#'}).count();
                    out.push(Section(n as u8, line.chars().skip(n).collect::<String>()));
                }
            }
        }
        */

        // More traditional + general scanner based
        let mut strm = Stream{s:txt, i:0};
        for i in 0..999999 {
            if strm.i >= strm.s.len() { break; }
            // println!("loop at i {} chr {:?}", strm.i, strm.s[strm.i..].chars().next().unwrap());

            let start_i = strm.i;

            if let Some(tok) = try_meta(&mut strm) {
                out.push(tok);
                continue;
            }

            if let Some(tok) = try_section(&mut strm) {
                if start_i != 0 { out.push(Tok::Break()); }
                out.push(tok);
                continue;
            }

            if let Some(tok) = try_mathblock(&mut strm) {
                out.push(tok);
                continue;
            }

            if let Some(tok) = try_code(&mut strm) {
                out.push(tok);
                continue;
            }

            if let Some(tok) = try_paragraph(&mut strm) {
                out.push(tok);
                continue;
            }

            /*
            if let Some(tok) = try_ticked(&mut strm) {
                out.push(tok);
                continue;
            }

            if let Some(tok) = try_link_or_image(&mut strm) {
                out.push(tok);
                continue;
            }

            if let Some(tok) = try_normal(&mut strm) {
                out.push(tok);
                continue;
            }
            */


            if let Some(tok) = try_double_newline(&mut strm) {
                out.push(tok);
                continue;
            }
            if let Some(tok) = try_newline(&mut strm) {
                // Do not push.
                continue;
            }

            panic!("some tok not hit (next char {})", &strm.s[strm.i..strm.i+1]);

        }

        return out;
    }

}

fn lower_doc_to_html(tokens: &Vec<Tok>, sh: Option<&SyntaxHighlighter>, top: bool) -> String {
    let mut html = String::new();

    for tok in tokens {
        use Tok::*;
        match tok {

            Ticked(s) => html += &format!("<span class=\"ticked\">{}</span>", s),
            TripleTicked(lang,code) => {
                if sh.is_some() {
                    html += &format!("<div class=\"code\">{}</div>", sh.unwrap().colorize(code,lang))
                } else {
                    html += &format!("<div class=\"code\">{}</div>", code)
                }
            },
            Underscored(s) => html += &format!("<span class=\"uscore\">{}</span>", s),
            OneStarred(s) => html += &format!("<span class=\"star1\">{}</span>", s),
            TwoStarred(s) => html += &format!("<span class=\"star2\">{}</span>", s),
            Link(label, link) => html += &format!("<a href=\"{}\">{}</a>", link, label),

            // Image(label, link) => html += &format!("<image src=\"{}\">{}</image>", link, label),
            Image(label, link) => {
                html += &format!("<figure>\n");
                html += &format!("\t<image src=\"{}\"></image>\n", link);
                html += &format!("\t<figcaption>{}<figcaption>\n", label);
                html += &format!("</figure>");
            }

            // Normal(s) => html += &format!("<p>{}</p>", s),
            Normal(s) => html += &format!("{}", s),
            Break() => html += "<br>\n",

            // FIXME: Ul, Ol (recursive)

            // Section(n,name) => html += &format!("<h{}>{}</h{}>", n, name, n),
            Section(n,inners) => {
                html += &format!("<h{}>", n);
                html += &lower_doc_to_html(inners, None, false);
                html += &format!("</h{}>", n);
            }

            Paragraph(inners) => {
                html += "<p>";
                html += &lower_doc_to_html(inners, None, false);
                html += "</p>";
            }

            Ul(_) => {}
            Ol(_) => {}

            // Noop
            Metadata(_) => {}

            // _ => { html += &format!("warning, unhandled token\n")); }
        }
    }

    html
}

pub fn parse_markdown_document(path: &str, sh: Option<&SyntaxHighlighter>) -> ParsedMarkdownDocument {
    // Read file.
    use std::fs::File;
    use std::io::Read;
    let mut fp = std::fs::OpenOptions::new()
                            .read(true)
                            .open(path).expect(&format!("failed to open input .md file file {}", path));
    let mut src = String::new();
    fp.read_to_string(&mut src);

    // Construct parser.
    let mdp = MarkdownParser {sh: sh};

    // Parse the document.
    let toks = mdp.lex(&src);


    // Create meta, if any 'Metadata' token exists.
    let mut meta = BTreeMap::new();
    for tok in &toks {
        if let Tok::Metadata(kvs) = tok {
            for (k,v) in kvs {
                meta.insert(k.clone(),v.clone());
            }
        }
    }

    // Lower to html.
    let html = lower_doc_to_html(&toks, sh, true);

    // println!("\nTokens:"); toks.into_iter().map(|t| println!("{:?}",t)).for_each(drop); println!("");
    // println!("\nHtml:\n{}", html);

    return ParsedMarkdownDocument {
        meta: meta,
        html: html,
        src_file: Some(path.to_owned()),
    };
}

#[test]
fn test_md() {
    let src = r#"# SectionA
## SectionB
`ticked`    `second ticked`
`not
ticked`
hello.
link to ![an image](/a/s/d)
`ticked followed by` [link](http)

A paragraph below.
Not a [](link
         )

```c
int main() {};
string s = "``";
```
#### After tt

"#;

    let mdp = MarkdownParser {sh: None};
    let toks = mdp.lex(&src);
    // println!("\nTokens:");
    // toks.into_iter().map(|t| println!("{:?}",t)).for_each(drop);
    // println!("");

    let s = "hello world!!";
    let mut strm = Stream{s:s, i:0};
    {
        let u1 = StreamUse::from(&mut strm);
    }
    {
        let u2 = StreamUse::from(&mut strm);
    }

}
