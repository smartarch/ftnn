package trust40.enforcer.sdq.io;



import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;


/**
 *
 */
public abstract class CSVLoader<T> {
    private final String PATH_PRIVACY_LEVEL;
    private static final char SEPERATOR_CHAR = ';';

    public CSVLoader(String path) {
        PATH_PRIVACY_LEVEL = path;
    }

    
    /**
     * Method to load
     *
     * @param coloumns
     * @return
     */
    protected String[][] loadCSVFile(int coloumns) throws IOException {
        if (coloumns < 1)
            throw new IllegalArgumentException("The number of coloumns needs to be bigger than 1");
        return Files.lines(Paths.get(PATH_PRIVACY_LEVEL)).map(e -> e.split(SEPERATOR_CHAR + ""))
                .toArray(String[][]::new);
    }


}
