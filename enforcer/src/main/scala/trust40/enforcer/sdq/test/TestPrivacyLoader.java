package trust40.enforcer.sdq.test;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import trust40.enforcer.sdq.io.PrivacyLoader;

class TestPrivacyLoader {
	static Path t;

	@BeforeAll
	static void init() {
		t = Paths.get("test.csv");
		try (BufferedWriter writer = Files.newBufferedWriter(t, Charset.forName("UTF-8"))) {
			writer.write("foreman;read(*);worker;public\n");
			writer.write("foreman;read(*);machine;sensitive\n");
            writer.write("worker;read(*);machine;highly_sensitive\n");
		} catch (IOException e) {
			e.printStackTrace();
			fail("IO-Error");
		}
	}
	@Test
	void testPrivacyTableCreation() {
		try {
			boolean test = TestHelper.getPrivacyTable().equals(new PrivacyLoader(t.toString()).getPrivacyTable());
            assertEquals(TestHelper.getPrivacyTable(),new PrivacyLoader(t.toString()).getPrivacyTable(),"Tests Privacy Table");
		} catch (IOException e) {
			e.printStackTrace();
			fail("IO-Error");
		}

	}
	@AfterAll
	static void clean() {
		try {
			Files.delete(t);
		} catch (IOException e) {
			e.printStackTrace();
			fail("IO-Error: can't delete privacy File");
		}
	}
}
