module.exports = {
  pathPrefix: '/steplee.github.io'
  siteMetadata: {
    title: `Gatsby Default Starter`,
  },
  plugins: [
    'gatsby-plugin-catch-links',
    'gatsby-plugin-react-helmet',

    // Markdown loading.
    {
      resolve: 'gatsby-source-filesystem',
      options: {
        path: 'src/posts',
        name: 'posts'
      }
    },

    // Markdown transforming.
    {
      resolve: 'gatsby-transformer-remark',
      options: {
        // Syntax-highlighting
        plugins: ['gatsby-remark-prismjs']
      }
    }
  ],
}
