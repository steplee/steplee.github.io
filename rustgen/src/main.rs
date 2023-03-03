
use cc;
mod syntax_highlight;
mod markdown_parser;

fn write_file(path: &str, data: &str) {
        use std::io::copy;
        use std::io::Write;
        use std::fs::File;
        let mut fp = std::fs::OpenOptions::new()
                                .write(true)
                                .create(true)
                                .truncate(true)
                                .open(path).expect(&format!("failed to open file {}", path));
        fp.write_all(data.as_bytes());
}


fn main() {
    // println!("Hello, world!");


    // Testing syntax highlighting.
    let code = r#"
    #define x 1
std::string s = "hello";
    char c = '1';
static int w;
struct Foo{};
int wtice(int x) { if(1) {Foo x;} return x.a().y(2,2).z % 2.; } // com
        "#;

        let sh = syntax_highlight::SyntaxHighlighter::new();
        let code_div = sh.colorize(&code, "cpp");

        let extra_part = r#"
<p>This is a paragraph.
It has a newline but no br.
Again.</p>

<p>This is a paragraph.<br>
It has a newline AND a br.<br>
Again.</p>

"#;


    // Testing markdown
    // let mdd = markdown_parser::parse_markdown_document("../posts/22_07_14_android_usb.md", None);
    let mdd = markdown_parser::parse_markdown_document("../posts/22_10_06_pytorch_autodiff_bundle_adjustment.md", None);

    let html = format!("<html><head><style>{}</style></head></body>\n{}\n{}\n{}\n</body></html>", sh.shared_stylesheet, code_div, extra_part, mdd.html);

    write_file("tmp.html", &html);


    /*
    download_treesitter_sources("tmp");

    // let dir: PathBuf = ["tree-sitter-javascript", "src"].iter().collect();
    let dir: PathBuf = ["tmp"].iter().collect();

    cc::Build::new()
        .include(&dir)
        .target("x86_64-unknown-linux-gnu")
        .host("x86_64-unknown-linux-gnu")
        .out_dir("tmp")
        .opt_level(3)
        .file(dir.join("parser.c"))
        .file(dir.join("scanner.c"))
        .compile("tree-sitter-javascript");
    */

}
