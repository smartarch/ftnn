name := "ftnn-simulator"

version := "1.0"

scalaVersion := "2.13.5"

mainClass in (Compile, run) := Some("ftnn.Main")

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-reflect" % "2.13.5",
  "org.scala-lang.modules" %% "scala-xml" % "1.3.0",
  "org.choco-solver" % "choco-solver" % "4.10.6",
  "com.typesafe.akka" %% "akka-actor" % "2.6.13",
  "com.typesafe.akka" %% "akka-stream" % "2.6.13",
  "com.typesafe.akka" %% "akka-http" % "10.2.4",
  "com.typesafe.akka" %% "akka-http-spray-json" % "10.2.4",
  "org.junit.jupiter" % "junit-jupiter-api" % "5.7.1",
  "org.junit.jupiter" % "junit-jupiter-engine" % "5.7.1",
  "org.hamcrest" % "hamcrest-all" % "1.3"
)
