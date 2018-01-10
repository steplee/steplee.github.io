/**
 * Implement Gatsby's Node APIs in this file.
 *
 * See: https://www.gatsbyjs.org/docs/node-apis/
 */


const path = require("path");

// Exported functions here will be run automatically by Gatsby

// uses Redux terminology -- "Bound Action Creators" 
// graphql is a *callback function*
exports.createPages = ({ boundActionCreators, graphql }) => {
  const { createPage } = boundActionCreators;

  const blogPostTemplate = path.resolve(`src/templates/blog-post.js`);

  return graphql(`{
    allMarkdownRemark(
        limit: 100
    ) {
      edges {
        node {
          excerpt(pruneLength: 250)
          html
          id
          frontmatter {
            date
            path
            title
          }
        }
      }
    }
  }`).then(result => {
    if (result.errors) {
      console.log("Error:" + result.errors);
      return Promise.reject(result.errors);
    }

    // Actually create the pages
    result.data.allMarkdownRemark.edges
      .forEach(({node}) => {
        console.log("Making page " + node.frontmatter.path);
        createPage({
          path: node.frontmatter.path,
          component: blogPostTemplate,
          context: {}
        });
      });
  });
};
