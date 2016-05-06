--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import           Data.Monoid (mappend)
import           Hakyll
import Hakyll.Web.Feed
import Control.Monad

--------------------------------------------------------------------------------

myFeedConfiguration = FeedConfiguration
    { feedTitle       = "steplee - Blog"
    , feedDescription = "the greatest tech blog ever"
    , feedAuthorName  = "Stephen Lee"
    , feedAuthorEmail = "stephenl7797@gmail.com"
    , feedRoot        = "http://steplee.github.io"
    }


--------------------------------------------------------------------------------

main :: IO ()
main =  do
  putStrLn "start"
  sequence_ [hakyll_main, putStrLn "done"]
  return ()

--------------------------------------------------------------------------------
hakyll_main =  hakyll $ do
    match "images/*" $ do
        route   idRoute
        compile copyFileCompiler
    match "images/icons/*" $ do
        route   idRoute
        compile copyFileCompiler

    match "js/*" $ do
        route   idRoute
        compile copyFileCompiler

    match "css/*.css" $ do
        route   idRoute
        compile compressCssCompiler

    match "css/*.scss" $ do
        route   $ setExtension "css"
        compile $ do
          rs <- getResourceString
          bdy <- withItemBody (unixFilter "sass" ["-s", "--scss"]) rs
          return $ fmap compressCss bdy

    match (fromList ["about.rst", "contact.markdown"]) $ do
        route   $ setExtension "html"
        compile $ pandocCompiler
            >>= loadAndApplyTemplate "templates/default.html" defaultContext
            >>= relativizeUrls

    match "posts/*" $ do
        route $ setExtension "html"
        compile $ pandocCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    create ["archive.html"] $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let archiveCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Archives"            `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/archive.html" archiveCtx
                >>= loadAndApplyTemplate "templates/default.html" archiveCtx
                >>= relativizeUrls


    match "index.html" $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let indexCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" ""                `mappend`
                    defaultContext

            getResourceBody
                >>= applyAsTemplate indexCtx
                >>= loadAndApplyTemplate "templates/index-default.html" indexCtx
                >>= relativizeUrls

    match "templates/*" $ compile templateBodyCompiler


------------------------- R S S -------------------------------

{-
    create ["atom.xml"] $ do
        route idRoute
        compile $ do
            let feedCtx = postCtx `mappend` bodyField "description"
            posts <- fmap (take 10) . recentFirst =<<
                loadAllSnapshots "posts/*" "content"
            renderAtom myFeedConfiguration feedCtx posts
-}


--------------------------------------------------------------------------------
postCtx :: Context String
postCtx =
    dateField "date" "%B %e, %Y" `mappend`
    defaultContext
