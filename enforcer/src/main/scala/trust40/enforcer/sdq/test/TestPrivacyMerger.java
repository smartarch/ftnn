package trust40.enforcer.sdq.test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;


import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import trust40.enforcer.sdq.DesignTimeDecisionMaker;
import trust40.enforcer.sdq.DesignTimeDecisionMakerImpl;
import trust40.enforcer.sdq.data.PrivacyLevel;
import trust40.enforcer.sdq.data.rules.AllowRule;
import trust40.enforcer.sdq.data.rules.DenyRule;
import trust40.k4case.AllowPermission;
import trust40.k4case.DenyPermission;
import trust40.k4case.Permission;

class TestPrivacyMerger {
	static DesignTimeDecisionMaker decision;
	static Path privacyConfig;
	static Path mappingConfig;

	@BeforeAll
	static void init() {
		privacyConfig = Paths.get("test.csv");
		mappingConfig = Paths.get("mapping.csv");
		try (BufferedWriter writerPrivacy = Files.newBufferedWriter(privacyConfig, Charset.forName("UTF-8"))) {

			writerPrivacy.write("foreman;read(*);machine;sensitive\n");
			writerPrivacy.write("worker;read(*);machine;highly_sensitive\n");
			writerPrivacy.write("foreman;read(*);worker;sensitive\n");
			writerPrivacy.close();
			BufferedWriter writerConfig = Files.newBufferedWriter(mappingConfig, Charset.forName("UTF-8"));
			writerConfig.write("A-foreman;foreman\n");
			writerConfig.write("A-worker-001;worker\n");
			writerConfig.write("A-worker-002;worker\n");
			writerConfig.write("factory;factory\n");
			writerConfig.write("dispenser;dispenser\n");
			writerConfig.write("machine-A;machine\n");
			writerConfig.close();
			decision = new DesignTimeDecisionMakerImpl(privacyConfig.getFileName().toString(),mappingConfig.getFileName().toString());
		} catch (IOException e) {
			e.printStackTrace();
			fail("IO-Error");
		}
	}

	@Test
	void testNoDenyRules() {
		List<AllowPermission> listAllow = createDefaultAllowPermission();
		List<DenyPermission> listDeny = new ArrayList<>();
		Map<AllowPermission,List<DenyPermission>> map = new HashMap<>();
		List<AllowPermission> permissions = decision.validatePolicies(listAllow,listDeny,map);
		
		assertThat(listAllow, is(permissions) );
	}
	@Test
	void testDenyRulesNotApplicable(){
		List<AllowPermission> listAllow = createDefaultAllowPermission();
		List<DenyPermission> listDeny = Stream.of(new DenyPermission("A-foreman", "read(test)", "A-worker-001", trust40.enforcer.tcof.PrivacyLevel.SENSITIVE()))
				.collect(Collectors.toList());
		Map<AllowPermission,List<DenyPermission>> map = new HashMap<>();
		List<AllowPermission> permissions = decision.validatePolicies(listAllow,listDeny,map);

		assertThat(listAllow, is(permissions) );
	}
	@Test
	void testDenyRulesWrongPrivacy(){
		List<AllowPermission> listAllow = createDefaultAllowPermission();
		listAllow.add(new AllowPermission("A-foreman", "read(phoneNumber)", "A-worker-001"));
		List<DenyPermission> listDeny = Stream.of(new DenyPermission("A-foreman", "read(*)", "A-worker-001", trust40.enforcer.tcof.PrivacyLevel.HIGHLY_SENSITIVE()))
				.collect(Collectors.toList());
		Map<AllowPermission,List<DenyPermission>> map = new HashMap<>();
		List<AllowPermission> permissions = decision.validatePolicies(listAllow,listDeny,map);
		assertThat(listAllow, is(permissions) );
	}
	@Test
	void testDenyRulesExactPrivacy(){
		List<AllowPermission> listAllow = createDefaultAllowPermission();
		listAllow.add(new AllowPermission("A-foreman", "read(phoneNumber)", "A-worker-001"));
		List<DenyPermission> listDeny = Stream.of(new DenyPermission("A-foreman", "read(*)", "A-worker-001", trust40.enforcer.tcof.PrivacyLevel.SENSITIVE()))
				.collect(Collectors.toList());
		Map<AllowPermission,List<DenyPermission>> map = new HashMap<>();
		List<AllowPermission> permissions = decision.validatePolicies(listAllow,listDeny,map);
		assertThat(createDefaultAllowPermission(), is(permissions) );
	}
	@Test
	void testDenyRulesLowerPrivacy(){
		List<AllowPermission> listAllow = createDefaultAllowPermission();
		listAllow.add(new AllowPermission("A-foreman", "read(phoneNumber)", "A-worker-001"));
		List<DenyPermission> listDeny = Stream.of(new DenyPermission("A-foreman", "read(*)", "A-worker-001", trust40.enforcer.tcof.PrivacyLevel.PUBLIC()))
				.collect(Collectors.toList());
		Map<AllowPermission,List<DenyPermission>> map = new HashMap<>();
		List<AllowPermission> permissions = decision.validatePolicies(listAllow,listDeny,map);
		assertThat(createDefaultAllowPermission(), is(permissions) );
	}
	private List<AllowPermission> createDefaultAllowPermission(){
		List<AllowPermission> listAllow = Stream.of(new AllowPermission("A-foreman", "enter()", "factory"),
				new AllowPermission("A-worker-001", "enter()", "dispenser"), new AllowPermission("A-worker-002", "read(machineData)", "machine-A"),
				new AllowPermission("A-foreman", "read(machineData)", "machine-A")
				)
				.collect(Collectors.toList());
		return listAllow;
	}

	@AfterAll
	static void clean() {
		try {
			Files.delete(privacyConfig);
			Files.delete(mappingConfig);
		} catch (IOException e) {
			e.printStackTrace();
			fail("IO-Error: can't delete privacy File");
		}
	}
}
